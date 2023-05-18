package org.apache.spark.ml.made

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.google.common.io.Files
import breeze.linalg.{*, DenseMatrix, DenseVector}

class LinearRegressionTest extends AnyFlatSpec with Matchers with WithSpark {

  val epsilon: Double = 0.01
  val modelWeights: DenseVector[Double] = LinearRegressionTest._modelWeights
  val modelBias: Double = LinearRegressionTest._modelBias
  val expectedOutput: DenseVector[Double] = LinearRegressionTest._expectedOutput

  val inputDataFrame: DataFrame = LinearRegressionTest._inputDataFrame

  private def validateModelPredictions(predictedModel: LinearRegModel, inputDataset: DataFrame): Unit = {

    val modelPredictions = inputDataset.collect().map(_.getAs[Double](1))

    modelPredictions.length should be (10000)
    for (i <- 0 until modelPredictions.length - 1) {
      modelPredictions(i) should be (expectedOutput(i) +- epsilon)
    }

  }

  private def validateModelParameters(estimatedModel: LinearRegModel): Unit = {

    val modelParameters = estimatedModel.weights

    modelParameters.size should be(modelWeights.size)
    modelParameters(0) should be (modelWeights(0) +- epsilon)
    modelParameters(1) should be (modelWeights(1) +- epsilon)
    modelParameters(2) should be (modelWeights(2) +- epsilon)
    estimatedModel.bias should be (modelBias +- epsilon)

  }

  "Estimator" should "calculate parameters" in {

    val estimator = new LinearReg()
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setMaxIter(100)
      .setStepSize(1.0)

    val estimatedModel = estimator.fit(inputDataFrame)

    validateModelParameters(estimatedModel)

  }

  "Model" should "make predictions" in {

    val predictedModel: LinearRegModel = new LinearRegModel(
      weights = Vectors.fromBreeze(modelWeights).toDense,
      bias = modelBias
    ).setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")

    validateModelPredictions(predictedModel, predictedModel.transform(inputDataFrame))

  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearReg()
        .setFeaturesCol("features")
        .setLabelCol("y")
        .setPredictionCol("prediction")
        .setMaxIter(100)
        .setStepSize(1.0)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reloadedModel = Pipeline.load(tmpFolder.getAbsolutePath)
      .fit(inputDataFrame).stages(0).asInstanceOf[LinearRegModel]

    validateModelParameters(reloadedModel)

  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearReg()
        .setFeaturesCol("features")
        .setLabelCol("y")
        .setPredictionCol("prediction")
        .setMaxIter(100)
        .setStepSize(1.0)
    ))

    val originalModel = pipeline.fit(inputDataFrame)

    val tmpFolder = Files.createTempDir()

    originalModel.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reloadedPipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModelPredictions(originalModel.stages(0).asInstanceOf[LinearRegModel], reloadedPipelineModel.transform(inputDataFrame))

  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _inputData: DenseMatrix[Double] = DenseMatrix.rand(10000, 3)
  lazy val _modelWeights: DenseVector[Double] = DenseVector(0.5, -0.1, 0.2)
  lazy val _modelBias: Double = 1.2
  lazy val _expectedOutput: DenseVector[Double] = _inputData * _modelWeights + _modelBias + DenseVector.rand(10000) * 0.0001

  lazy val _inputDataFrame: DataFrame = constructDataFrame(_inputData, _expectedOutput)

  def constructDataFrame(inputData: DenseMatrix[Double], outputData: DenseVector[Double]): DataFrame = {

    import sqlc.implicits._

    lazy val combinedData: DenseMatrix[Double] = DenseMatrix.horzcat(inputData, outputData.asDenseMatrix.t)

    lazy val df = combinedData(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq.toDF("feature1", "feature2", "feature3", "y")

    lazy val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("feature1", "feature2", "feature3"))
      .setOutputCol("features")

    lazy val _inputDataFrame: DataFrame = vectorAssembler
      .transform(df)
      .select("features", "y")

    _inputDataFrame
  }

}
