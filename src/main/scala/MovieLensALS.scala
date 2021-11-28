import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import scala.util.Random

object MovieLensALS {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    // unpack tuple returned by parseArgs
    val (ratingsPath, doGrading) = parseArgs(args)
    val load_movie_preferences = true
    // set up environment

    val conf = new SparkConf()
      .setAppName("MovieLensALS")
//      .setMaster("local[2]")
    val sc = new SparkContext(conf)

    // Read dataset into memory. For more information, view comments i15948n loadFilms and loadRatings
    val filmId2Title = loadFilms(ratingsPath + "/movies2.csv", sc)
    val initRatingsData = loadRatings(ratingsPath + "/ratings2.csv", sc)

    // extra filtering - exclude movies that appear less than 50 times
    val rareMoviesId = initRatingsData.map(_.product).countByValue().filter(_._2 < 50).keys.toSet
    val ratingsData = initRatingsData.filter(x => !rareMoviesId.contains(x.product))

    //    val films = Random.shuffle(
    //      sc
    //        .textFile(ratingsPath + "/for-grading.tsv")
    //        .map{_.split("\t")}
    //        .map{x => (x(0).toInt, x(1))}
    //        .filter(x => (rareMoviesId.contains(x._1)))
    //        .collect()
    //        .toSeq
    //    )
    //    println(films.length)

    // if user flag is set, survey user for movie preferences
    val myRating = if (doGrading)
      (new Grader(ratingsPath, load_movie_preferences, sc)).toRDD
    else
    // else create user with no preferences
      sc.parallelize(Seq[Rating]())
    val Array(train, test) = ratingsData.randomSplit(Array(0.9, 0.1), seed=42L)
    val baseline = makeBaseline(ratingsData)

    // hyper-parameter RANK fine-tuning.
    // Commeneted as not needed for a single train/test mode
    // train a matrix factorization model
    // user should be in training data split. otherwise preference prediction is impossible
    // two parameters are rank (specifies complexity) and iterations

//    val ranks = Array(60)
      var best_params = Array(10, Double.PositiveInfinity)
//
//    for( r <- ranks ) {
//      println(s"Rank: ${r}")
//      val model = ALS.train(train.union(myRating), r, 10)
//
//      // https://spark.apache.org/docs/2.3.2/api/java/org/apache/spark/mllib/recommendation/MatrixFactorizationModel.html#predict-org.apache.spark.rdd.RDD-
//      val prediction = model.predict(test.map(x => (x.user, x.product)))
//
//      // calculate validation error
//      val recommenderRmse = rmse(test, prediction)
//      println(s"Error after training: ${recommenderRmse}")
//
//      // if baseline is implemented, calculate baseline error
//      if (baseline.nonEmpty) {
//        val baselineRmse = rmse(test, baseline)
//        println(s"Baseline error: ${baselineRmse}")
//      }
//
//      if (recommenderRmse < best_params(1)) {
//        best_params = Array(r, recommenderRmse)
//      }
//      println("\n")
//    }

    // the best model
    val model = ALS.train(train.union(myRating), best_params(0).toInt, 10)

    //
    //    // https://spark.apache.org/docs/2.3.2/api/java/org/apache/spark/mllib/recommendation/MatrixFactorizationModel.html#predict-org.apache.spark.rdd.RDD-
    val prediction = model.predict(test.map(x => (x.user, x.product)))
    //prediction.foreach(println)

    // calculate validation error
    val recommenderRmse = rmse(test, prediction)
    println(s"Error after training: ${recommenderRmse}")
    //
    //    // if baseline is implemented, calculate baseline error
    if (baseline.nonEmpty) {
      val baselineRmse = rmse(test, baseline)
      println(s"Baseline error: ${baselineRmse}")
    }

    println("\n\n")

    // if user preferences are specified, predict top 20 movies for the user
    // default user id = 0. the same user id is set in Grader.scala
    if (doGrading){
      println("Predictions for user with filtering\n")
      val already_watched = myRating.map(x => x.product).collect()
      // for input data format refer to documentation
      // get movie ids to rank from baseline map
      model.predict(sc.parallelize(baseline.keys.map(x => (0, x)).toSeq))
        // sort by ratings in descending order
        .sortBy(_.rating, ascending = false)
        // filter non-watched
        .filter(x => !already_watched.contains(x.product))
        // take first 20 items
        .take(20)
        // print title and predicted rating
        .foreach(x => println(s"${filmId2Title(x.product)}\t\t${x.rating}"))
    }

    sc.stop()
  }


  // return types are inferred automatically
  def parseArgs(args: Array[String]) = {

    // instances instantiated with val are immutable
    val usageString = "\n\nUsage: path/to/spark-submit path/to/jar movie/lens/data/path -user [true/false]\n\n"

    // instances instantiated with var can be modified
    var doGrading = false

    // first array element fetched using round brackets
    val ratingsPath = args(0)

    var proceed = true

    if (args.length != 3) {
      proceed = false
    }
    if (args(1) == "-user")
      try{
        doGrading = args(2).toBoolean
      } catch {
        case e: Exception => proceed = false
      }

    if (!proceed) {
      println(usageString)
      sys.exit(1)
    }

    // return tuple
    (ratingsPath, doGrading)
  }



  def loadRatings(path: String, sc: SparkContext) = {

    // the file of interest (that contains ratings) is located in HDFS
    // files from HDFS are read with SparkContext's method textFile
    // by default textFile return RDD[String] ([] - specify template parameters)
    val ratingsData = sc.textFile(path).map{ line =>
      val fields = line.split(",")
      // file data format is
      // userID, movieID, movieRating, time
      // our recommendation system is trained on data of type RDD[Rating]
      // https://spark.apache.org/docs/2.3.2/api/java/org/apache/spark/mllib/recommendation/Rating.html
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }
    ratingsData
  }


  def loadFilms(path: String, sc: SparkContext) = {

    // read the file with movie info from HDFS
    // Movie information is contained in the file `movies.csv`.
    // Each line of this file after the header row represents one movie,
    // and has the following format:
    //      movieId,title,genres
    // Columns that contain commas (`,`) are escaped using double-quotes (`"`).
    // Example :
    //      39,Clueless (1995),Comedy|Romance
    //      40,"Cry, the Beloved Country (1995)",Drama

    sc.textFile(path).map { line =>
      val id = parseId(line)
      val title = parseTitle(line)
      (id, title)
    } // create map (similar to HashMap in Java)
      .collectAsMap()
  }

  def makeBaseline(ratingsData: RDD[Rating]) = {

    // calculate baseline for movie ratings
    // let's agree that the baseline is the average rating for the movie
    // for every movie we need to collect all the ratings and find their average
    // we need to transform the data in the format
    // movieId: [rating1, rating2, ..., ratingN]
    // creating these arrays is memory intensive
    // alternatively, use reduce by key

    ratingsData
      // transform data in the format (movieId, (rating, 1))
      // 1 is needed to count the number of ratings
      .map(x => (x.product, (x.rating, 1:Int)))
      // reduce by key
      // https://spark.apache.org/docs/latest/rdd-programming-guide.html#working-with-key-value-pairs
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
      // now you have (movieId, (ratingSum, ratingCount))
      // transform it into (movieId, averageRating)
      .map(x => (x._1, x._2._1 / x._2._2))
      // create map (similar to HashMap in Java)
      .collectAsMap()
  }

  def parseId(filmEntry: String) = {
    // more info about regex in scala and
    // findFirstIn in particular refer to
    // https://www.scala-lang.org/api/2.12.5/scala/util/matching/Regex.html
    "^[0-9]*,".r findFirstIn filmEntry match {
      // match is an equivalent of case switch
      // https://docs.scala-lang.org/tour/pattern-matching.html

      // what is Some() and how to use it
      // https://alvinalexander.com/scala/using-scala-option-some-none-idiom-function-java-null
      case Some(s) => s.slice(0, s.length - 1).toInt
      case None => throw new Exception(s"Cannot parse Id in {$filmEntry}")
    }
  }


  def parseTitle(filmEntry: String) = {
    val splitted_line = filmEntry.split(",")
    if (splitted_line.length > 3) {
      "\"(.*)\"".r findFirstIn filmEntry match {
        case Some(s) => s.slice(1, s.length - 1)
        case None => throw new Exception(s"Cannot parse Film Title in {$filmEntry}")
      }
    }
    else {
      splitted_line(1)
    }
  }


  def rmse(test: RDD[Rating], prediction: RDD[Rating]) = {
    val rating_pairs = prediction
      .map(x => ((x.user, x.product), x.rating))
      // more about join
      // https://spark.apache.org/docs/2.1.1/api/java/org/apache/spark/rdd/PairRDDFunctions.html#join(org.apache.spark.rdd.RDD)
      .join(test.map(x => ((x.user, x.product), x.rating)))

    math.sqrt(rating_pairs.values
      .map(x => (x._1 - x._2) * (x._1 - x._2))
      // _ + _ is equivalent to the lambda (x,y) => x+y
      .reduce(_ + _) / test.count())
  }

  def rmse(test: RDD[Rating], prediction: scala.collection.Map[Int, Double]) = {
    val rating_pairs = test.map(x => (prediction.get(x.product), x.rating)).filter(_._1.isDefined).map(x => (x._1.get, x._2))
    math.sqrt(rating_pairs
      .map(x => (x._1 - x._2) * (x._1 - x._2))
      // _ + _ is equivalent to the lambda (x,y) => x+y
      .reduce(_ + _) / test.count())
  }
}
