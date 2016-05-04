/**
 * @author Saket Vishwasrao
 * This is the code to cluster using TF-IDF as feature extraction.
 * 
 * Tuning parameters:
 * minDocFreq for IDF scores
 * numClusters: Number of clusters for input to clustering algorithm
 * 
 */


import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.{KMeans,KMeansModel}
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.feature.Normalizer

object DirectClustering {
  
   def main(args: Array[String]) {
     
   val sc = new SparkContext("local[2]", "Clustering")
   val users = sc.textFile(args(0))

   users.first()
 /************Preprocessing********************/
  
   val filtereddata = users.filter(!_.isEmpty).map(line=>line.split("\t",-1)).map(line=>(line(0),line(1).split(" ").filter(_.nonEmpty)))
   
   
 
   val values:RDD[Seq[String]] = filtereddata.map(s=>s._2)
   val keys = filtereddata.map(s=>s._1)
   println(values.take(3).mkString("\n"))
   
/*******************TF-IDF and normalisation*****************************/
   
   val hashingTF:RDD[Vector] = new HashingTF().transform(values)
   val idf= new IDF(minDocFreq=10).fit(hashingTF)
   val tfidf:RDD[Vector]= idf.transform(hashingTF).cache()
   
   val normalizer =  new Normalizer()
   val data1= tfidf.map(x=>(normalizer.transform(x)))
   
/*********************Clustering Algorithm***********************************/   
   
   val clusters = KMeans.train(data1,5,10)
   val predictions=	clusters.predict(data1)
   val clustercount= predictions.map(s=>(s,1)).reduceByKey(_+_)
  // println(clustercount.take(4).mkString(" "))
/******************Compute metrics***********************************/
   val result= keys.zip(values).zip(predictions)
   val wsse = clusters.computeCost(data1)
  // println(result.take(4).mkString(" "))
   println(result.take(4).mkString(" "))
   println(clustercount.take(5).mkString(" "))
   println(wsse.toString)
   //result.saveAsTextFile(path)
   
   }
}