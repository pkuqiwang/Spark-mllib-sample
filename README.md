# Spark Machine learning sample using RDD and MLLib

## Setup environment 
Download Hortonworks HDP 2.4 [sandbox](https://hortonworks.com/downloads/#sandbox) and prepare the sandbox following this [instruction](http://hortonworks.com/hadoop-tutorial/learning-the-ropes-of-the-hortonworks-sandbox/)  

After the sandbox is prepared, you should see in Ambari dashboard that Spark and Zeppelin Notebook both turns green

THen go to Zeppelin notebook using [http://127.0.0.1:9995](http://127.0.0.1:9995). 

Click Notebook -> create new notebook, name the notebook. 

##Prepare the flight delay dataset
### Download dataset
Download the dataset from internet
```
%sh
wget http://stat-computing.org/dataexpo/2009/2007.csv.bz2 -O /tmp/flights_2007.csv.bz2
wget http://stat-computing.org/dataexpo/2009/2008.csv.bz2 -O /tmp/flights_2008.csv.bz2
wget ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2007.csv.gz -O /tmp/weather_2007.csv.gz
wget ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2008.csv.gz -O /tmp/weather_2008.csv.gz
ls -l /tmp/
```
### Copy dataset to HDFS
```
%sh
#remove existing copies of dataset from HDFS
hdfs dfs -rm -r -f /tmp/airflightsdelays
hdfs dfs -mkdir /tmp/airflightsdelays

#put data into HDFS
hdfs dfs -put /tmp/flights_2007.csv.bz2 /tmp/flights_2008.csv.bz2 /tmp/airflightsdelays/
hdfs dfs -put /tmp/weather_2007.csv.gz /tmp/weather_2008.csv.gz /tmp/airflightsdelays/
hdfs dfs -ls -h /tmp/airflightsdelays
```
## Machine learning process
###Prepare the traning and testing data 
```
%spark
import org.apache.spark.mllib.regression.LabeledPoint

//calculate minuted from midnight, input is military time format 
def getMinuteOfDay(depTime: String) : Int = (depTime.toInt / 100).toInt * 60 + (depTime.toInt % 100)

val flight2007 = sc.textFile("/tmp/airflightsdelays/flights_2007.csv.bz2")
val header = flight2007.first

val trainingData = flight2007
                    .filter(x => x != header)
                    .map(x=>x.split(","))
                    .filter(x => x(21) == "0")
                    .filter(x => x(16) == "ORD")
                    .filter(x => x(14) != "NA")
                    .map(x => LabeledPoint(if (x(14).toInt >= 15) 1.0 else 0.0, Vectors.dense(x(1).toInt, x(2).toInt, x(3).toInt, getMinuteOfDay(x(4)), getMinuteOfDay(x(6)), x(11).toInt, x(15).toInt, x(18).toInt)))
trainingData.cache

val flight2008 = sc.textFile("/tmp/airflightsdelays/flights_2008.csv.bz2")
val testingData = flight2008
                    .filter(x => x != header)
                    .map(x=>x.split(","))
                    .filter(x => x(21) == "0")
                    .filter(x => x(16) == "ORD")
                    .filter(x => x(14) != "NA")
                    .map(x => LabeledPoint(if (x(14).toInt >= 15) 1.0 else 0.0, Vectors.dense(x(1).toInt, x(2).toInt, x(3).toInt, getMinuteOfDay(x(4)), getMinuteOfDay(x(6)), x(11).toInt, x(15).toInt, x(18).toInt)))
testingData.cache
```
###Train model with Decision Tree 
```
%spark
import org.apache.spark.mllib.tree.DecisionTree

// Build the Decision Tree model
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 10
val maxBins = 100
val modelDecisionTree = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

// Predict
val labelsAndPredsDecisionTree = testingData.map { point => 
    val pred = modelDecisionTree.predict(point.features) 
    (pred, point.label)
}

// Get evaluation metrics.
val metricslabelsAndPredsDecisionTree = new BinaryClassificationMetrics(labelsAndPredsDecisionTree)
val auROCmetricslabelsAndPredsDecisionTree = metricslabelsAndPredsDecisionTree.areaUnderROC()
```
###Train model with Random Forest 
```
%spark

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Strategy

val treeStrategy = Strategy.defaultStrategy("Classification")
val numTrees = 100 
val featureSubsetStrategy = "auto" // Let the algorithm choose
val model_rf = RandomForest.trainClassifier(trainingData, treeStrategy, numTrees, featureSubsetStrategy, seed = 123)

// Predict
val labelsAndPredsRandomForest = testingData.map { point =>
    val pred = model_rf.predict(point.features)
    (point.label, pred)
}

// Get evaluation metrics.
val metricslabelsAndPredsRandomForeste = new BinaryClassificationMetrics(labelsAndPredsRandomForest)
val auROCmetricslabelsAndPredsRandomForeste = metricslabelsAndPredsRandomForeste.areaUnderROC()
```

###Normalize data for regression algorithms
```
%spark
import org.apache.spark.mllib.feature.StandardScaler

val trainingScaler = new StandardScaler(withMean = true, withStd = true).fit(trainingData.map(x => x.features))
val scaledTrainingData = trainingData.map(x => LabeledPoint(x.label, trainingScaler.transform(Vectors.dense(x.features.toArray))))
scaledTrainingData.cache

val testingScaler = new StandardScaler(withMean = true, withStd = true).fit(testingData.map(x => x.features))
val scaledTestingData = testingData.map(x => LabeledPoint(x.label, testingScaler.transform(Vectors.dense(x.features.toArray))))
scaledTestingData.cache
```
###Train model with Logistic Regression 
```
%spark 
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

// Build the Logistic Regression model
val model_lr = LogisticRegressionWithSGD.train(scaledTrainingData, numIterations=10)

// Predict
val labelsAndPreds_lr = scaledTestingData.map { point =>
    val pred = model_lr.predict(point.features)
    (pred, point.label)
}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(labelsAndPreds_lr)
val auROC = metrics.areaUnderROC()
```

###Train model with SVM with SGD
```
%spark 
import org.apache.spark.mllib.classification.SVMWithSGD

// Build the SVM model
val model_svm = SVMWithSGD.train(scaledTrainingData, numIterations=10 )

// Predict
val labelsAndPreds_svm = scaledTestingData.map { point =>
        val pred = model_svm.predict(point.features)
        (pred, point.label)
}

// Get evaluation metrics.
val metricsSVMWithSGD = new BinaryClassificationMetrics(labelsAndPreds_svm)
val auROC = metricsSVMWithSGD.areaUnderROC()
```
