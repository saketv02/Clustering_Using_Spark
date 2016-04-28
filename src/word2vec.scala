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



object word2vec {
  
   def main(numClusters:Int,collection:String) {
   val sc = new SparkContext("local[2]", "Clustering")
   val users = sc.textFile("user/cs5604s16_sn/input/tweets/" + collection +".txt" )

   users.first()
   
   val filtereddata = users.filter(!_.isEmpty).map(line=>line.split("\t",-1)).map(line=>(line(0),line(1).split(" ").filter(_.nonEmpty)))
   val corpus =  users.filter(!_.isEmpty).map(line=>line.split("\t",-1)).map(line=>line(1).split(" ").toSeq)
   
   
   val values:RDD[Seq[String]] = filtereddata.map(s=>s._2.filter(_.nonEmpty))
   val keys = filtereddata.map(s=>s._1)
   
  
   println(corpus.take(1).mkString("\n"))
   /*
   val hashingTF:RDD[Vector] = new HashingTF().transform(values)
   val idf= new IDF(minDocFreq=10).fit(hashingTF)
   val tfidf:RDD[Vector]= idf.transform(hashingTF).cache()
   */
   def sumArray (m: Array[Double], n: Array[Double]): Array[Double] = {
		   	for (i <- 0 until m.length) {m(i) += n(i)}
		   	return m
   }

   def divArray (m: Array[Double], divisor: Double) : Array[Double] = {
		   for (i <- 0 until m.length) {m(i) /= divisor}
		   return m
   }

   def wordToVector (w:String, m: Word2VecModel): Vector = {
		   try {
			   return m.transform(w)
		   } catch {
		   case e: Exception => return Vectors.zeros(100)
		   }  
   }
   
   
   val vec = new Word2Vec()
   val model = vec.fit(corpus)
   
  // val test =  values.map(x=> new DenseVector(x.map(m=>wordToVector(m,model).toArray).reduceLeft(sumArray)).asInstanceOf[Vector])
  // test.count()
   val test =  values.map(x=> (x.map(m=>wordToVector(m,model).toArray))).filter(_.length!=0).map(x=> new DenseVector(divArray(x.reduce(sumArray),x.length)).asInstanceOf[Vector])
   
   val normalizer =  new Normalizer()
   val data1= test.map(x=>(normalizer.transform(x)))
   
   

  
    
   val clusters = KMeans.train(data1,numClusters,10)
   val predictions=	clusters.predict(data1)
   val clustercount= predictions.map(s=>(s,1)).reduceByKey(_+_)
   val result= keys.zip(values).zip(predictions)
   val wsse = clusters.computeCost(data1)
   
   
   result.saveAsTextFile("/user/cs5604s16_sn/output/tweets/"+collection)
   println(result.take(4).mkString(" "))
   println(clustercount.take(numClusters).mkString(" "))
   println(wsse.toString)
   
   
   }
}