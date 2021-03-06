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
   
   val filtereddata = users.filter(!_.isEmpty).map(line=>line.split("\t",-1)).map(line=>(line(0),line(1).split(" ").filter(_.nonEmpty)))
   
   val values:RDD[Seq[String]] = filtereddata.map(s=>s._2)
   val keys = filtereddata.map(s=>s._1)
   
   
   
   println(values.take(3).mkString("\n"))
   val hashingTF:RDD[Vector] = new HashingTF().transform(values)
   val idf= new IDF(minDocFreq=10).fit(hashingTF)
   val tfidf:RDD[Vector]= idf.transform(hashingTF).cache()
   
   val normalizer =  new Normalizer()
   val data1= tfidf.map(x=>(normalizer.transform(x)))
   
   
   val clusters = KMeans.train(data1,5,10)
   val predictions=	clusters.predict(data1)
   val clustercount= predictions.map(s=>(s,1)).reduceByKey(_+_)
  // println(clustercount.take(4).mkString(" "))
   val result= keys.zip(values).zip(predictions)
   val wsse = clusters.computeCost(data1)
  // println(result.take(4).mkString(" "))
   println(result.take(4).mkString(" "))
   println(clustercount.take(5).mkString(" "))
   println(wsse.toString)
   //result.saveAsTextFile(path)
   
   }
}