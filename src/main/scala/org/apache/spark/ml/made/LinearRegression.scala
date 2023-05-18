package org.apache.spark.ml.made

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, HasStepSize}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter, MetadataUtils, MLReadable, MLReader, MLWritable, MLWriter}

trait LRParams extends HasLabelCol with HasFeaturesCol with HasPredictionCol with HasMaxIter with HasStepSize {
  def setLabelCol(value: String) : this.type = set(labelCol, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  setDefault(maxIter -> 100, stepSize -> 0.1)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}

class LinearReg(override val uid: String) extends Estimator[LinearRegModel] with LRParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearReg"))

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setStepSize(value: Double): this.type = set(stepSize, value)

  override def fit(dataset: Dataset[_]): LinearRegModel = {
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()
    val featuresExt = dataset.withColumn("ones", lit(1))
    val assembler = new VectorAssembler()
      .setInputCols(Array($(featuresCol), "ones", $(labelCol)))
      .setOutputCol("features_ext")

    val assembledFeatures: Dataset[Vector] = assembler
      .transform(featuresExt)
      .select("features_ext").as[Vector]

    val numFeatures: Int = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var weights: breeze.linalg.DenseVector[Double] = breeze.linalg.DenseVector.rand[Double](numFeatures + 1)

    for (_ <- 0 to $(maxIter)) {
      val summary = assembledFeatures.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until weights.size).toDenseVector
          val y = v.asBreeze(weights.size)
          val grad = X * (breeze.linalg.sum(X * weights) - y)
          summarizer.add(mllib.linalg.Vectors.dense(grad.toArray), 1.0)
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      val avgGrad = summary.normL2(0) / summary.count
      weights -= $(stepSize) * avgGrad
    }

    copyValues(new LinearRegModel(weights = weights, uid = uid).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
  override def copy(extra: ParamMap): LinearReg = defaultCopy(extra)
}

object LinearReg extends DefaultParamsReadable[LinearReg]

class LinearRegModel private[made](override val uid: String, val weights: DenseVector[Double])
  extends Model[LinearRegModel] with LRParams with DefaultParamsWritable {

  override def copy(extra: ParamMap): LinearRegModel = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val broadcastedWeights = dataset.sparkSession.sparkContext.broadcast(weights)
    val transformUDF = dataset.sqlContext.udf.register(uid, (features: Vector) => {
      val X = Vectors.dense(features.toArray :+ 1.0)
      val prediction = X.asBreeze.dot(broadcastedWeights.value.asBreeze)
      Vectors.dense(prediction)
    })

    dataset.withColumn($(predictionCol), transformUDF(dataset($(featuresCol))))
  }
}

object LinearRegModel extends MLReadable[LinearRegModel] {
  override def read: MLReader[LinearRegModel] = new DefaultParamsReader[LinearRegModel]
  class LinearRegModelWriter(instance: LinearRegModel) extends MLWriter {
    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.uid, instance.weights)
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(path + "/data")
    }
    case class Data(uid: String, weights: Vector)
  }
  override def load(path: String): LinearRegModel = {
    val metadata = DefaultParamsReader.loadMetadata(path, sc)
    val data = sparkSession.read.parquet(path + "/data").select("weights").head()
    val weights = data.getAs[Vector](0).asBreeze.fromBreeze
    val model = new LinearRegModel(metadata.uid, weights)
    metadata.getAndSetParams(model)
    model
  }
}
