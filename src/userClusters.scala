/**
 * @author Saket Vishwasrao
 * This is the code to cluster users.
 * 
 * Tuning parameters:
 * MinDocFreq for IDF scores
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

object clustering {
  
   final private val path=" ";  //enter hdfs path here
  
   def main(numClusters:Int,collection:String) {
     
   val sc = new SparkContext("local[2]", "Clustering")  //comment this line if running inside spark-shell
   val users = sc.textFile(path)

   users.first()
   
/************Preprocessing********************/
   
   val userlist = users.filter(!_.isEmpty).map(line=>line.split(",")).map(array=>(array(3),array(1))).map(word=> (word._1,word._2.split(" ").filter(word=> word.startsWith("#"))))

   val cleanList = userlist.map(word=>(word._1.split(" ").filter(_.nonEmpty),word._2)).map(word=>(word._1(0),word._2))
  
   val reducedList = cleanList.reduceByKey(_++_)
   val arr = reducedList.first()._2
   reducedList.take(5).foreach(println);
   println(arr.deep.mkString(" "))
   
   val values:RDD[Seq[String]] = reducedList.map(s=>s._2)
   val keys = reducedList.map(s=>s._1)

/*******************TF-IDF*****************************/
   
   val hashingTF:RDD[Vector] = new HashingTF().transform(values).cache()
   val idf= new IDF(minDocFreq=1000).fit(hashingTF)
   val tfidf:RDD[Vector]= idf.transform(hashingTF).cache()
   
   println(values.take(2).mkString("\n"))
   println(hashingTF.take(2).mkString("\n"))
   println(tfidf.take(2).mkString("\n"))
   
/*********************Clustering Algorithm***********************************/   
   
   val clusters = KMeans.train(tfidf,numClusters,20)
   
/******************Compute metrics***********************************/
   
   val predictions=	clusters.predict(tfidf).map(s=>(s,1)).reduceByKey(_+_)
   println(predictions.take(numClusters).mkString(" "))
   
   val wsse =  clusters.computeCost(hashingTF)
   println("Within set sum of squares=" + wsse)
   
   }
}
