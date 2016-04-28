import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.{KMeans,KMeansModel}
import org.apache.spark.mllib.feature.IDF

object clustering {
  
   def main(args: Array[String]) {
   val sc = new SparkContext("local[2]", "Clustering")
   val users = sc.textFile(args(0))

   users.first()

   val userlist = users.filter(!_.isEmpty).map(line=>line.split(",")).map(array=>(array(3),array(1))).map(word=> (word._1,word._2.split(" ").filter(word=> word.startsWith("#"))))

   val cleanList = userlist.map(word=>(word._1.split(" ").filter(_.nonEmpty),word._2)).map(word=>(word._1(0),word._2))
  
  
   val reducedList = cleanList.reduceByKey(_++_)
   val arr = reducedList.first()._2
   reducedList.take(5).foreach(println);
   println(arr.deep.mkString(" "))
   
   val values:RDD[Seq[String]] = reducedList.map(s=>s._2)
   val keys = reducedList.map(s=>s._1)


   val hashingTF:RDD[Vector] = new HashingTF().transform(values).cache()
   val idf= new IDF(minDocFreq=5000).fit(hashingTF)
   val tfidf:RDD[Vector]= idf.transform(hashingTF).cache()
   
   println(values.take(2).mkString("\n"))
   println(hashingTF.take(2).mkString("\n"))
   println(tfidf.take(2).mkString("\n"))
   
   val clusters = KMeans.train(tfidf,4,20)
   val predictions=	clusters.predict(tfidf).map(s=>(s,1)).reduceByKey(_+_)
   println(predictions.take(4).mkString(" "))
   
   
  // println(predictions.take(30).mkString(" "))
   
  // val wsse =  clusters.computeCost(hashingTF)
  // println("Within set sum of squares=" + wsse)
   
   }
}
