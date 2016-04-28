import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.{KMeans,KMeansModel}
import org.apache.spark.mllib.feature.IDF





object webpage {

	def main(args: Array[String]) {
		val sc = new SparkContext("local[2]", "Clustering")
		val raw = sc.textFile(args(0))
		val webpages = raw.filter(!_.isEmpty()).map(line=>line.split("\t",-1)).map(line=>(line(0),line(1).split(" ").filter(_.nonEmpty)))
		webpages.first()
	}
}