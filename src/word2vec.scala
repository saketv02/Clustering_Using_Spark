import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.{KMeans,KMeansModel}
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.feature.{Word2Vec,Word2VecModel}
import org.apache.spark.mllib.feature.Normalizer
import scala.collection.immutable.StringLike


object word2vec {
  
  
    
   def main(numClusters:Int,collection:String) {
  // val sc = new SparkContext("local[2]", "Clustering")
   val users = sc.textFile("input/tweets" + collection +".txt" )

   users.first()
   
   val filtereddata = users.filter(!_.isEmpty).map(line=>line.split("\t",-1)).map(line=>(line(0),line(1).split(" ").filter(_.nonEmpty)))
   val corpus =  users.filter(!_.isEmpty).map(line=>line.split("\t",-1)).map(line=>line(1).split(" ").toSeq)
    
   
   val values:RDD[Seq[String]] = filtereddata.map(s=>s._2.filter(_.nonEmpty))
   
   val keys = filtereddata.map(s=>s._1)
   
  
   println(corpus.take(1).mkString("\n"))
  
  
   
   val vec = new Word2Vec()
   val model = vec.fit(corpus)
   println("fitting done......................................")
   println(values.take(2).mkString("\n"))
   
   val outtest:RDD[Seq[Vector]]= values.map(x=>x.map(m=>try {
		     model.transform(m)
		   } catch {  
		   case e: Exception => Vectors.zeros(100)//return Vectors.zeros(100)
		   }))
		   
  val convertest = outtest.map(m=>m.map(x=>(x.toArray)))
  val reducetest = convertest.map(x=>x.reduce((a,b)=>a.zip(b).map(t=>t._1+t._2)))
  
  val filtertest = reducetest.map(x=>x.map(m=>(m,x.length)).map(m=>m._1/m._2))
  
  val test = filtertest.map(x=>new DenseVector(x).asInstanceOf[Vector])
		   
		   
   println(test.take(10).mkString("\n"))
   val normalizer =  new Normalizer()
   val data1= test.map(x=>(normalizer.transform(x)))
   
   

  
    
   val clusters = KMeans.train(data1,numClusters,10)
   val predictions=	clusters.predict(data1)
   val clustercount= predictions.map(s=>(s,1)).reduceByKey(_+_)
   val result= keys.zip(values).zip(predictions)
   val wsse = clusters.computeCost(data1)
   
   
   result.saveAsTextFile("output/tweets"+collection)
   println(result.take(4).mkString(" "))
   println(clustercount.take(numClusters).mkString(" "))
   println(wsse.toString)
   
   
   }
}