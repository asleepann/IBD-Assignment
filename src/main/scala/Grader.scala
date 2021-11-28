import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import scala.util.Random

import java.io.File
import java.io.PrintWriter


class Grader(path: String, load_movie_preferences: Boolean, sc: SparkContext) {

  // constructor begins

  val films = Random.shuffle(
    sc
      .textFile(path + "/for-grading.tsv")
      .map {
        _.split("\t")
      }
      .map { x => (x(0).toInt, x(1)) }
      .collect()
      .toSeq
  )

  // Need to specify type when it cannot be inferred
  var graded: Seq[(Int, Double)] = Seq()

  if (load_movie_preferences) {
    graded = sc.textFile(path + "/user_ratings.tsv")
      .map(line => line.split("\t"))
      .map(field => (field(0).toInt, field(1).toDouble))
      .collect()
      .toSeq
  } else {

    var position = 0

    val grading_message = "Enter grade from 1 to 5. Enter 0 if you did not see this movie: "

    var step = 0
    var limit = 20


    while (step < limit) {

      println(s"\nGraded ${step}/${limit}, Viewed ${position}/${films.length}")
      println(s"${grading_message}\n${films(position)._2}")

      var grade = scala.io.StdIn.readLine()

      try {

        var intGrade = grade.toDouble

        if (intGrade < 0.0 || intGrade > 5.0) {
          throw new NumberFormatException()
        }

        if (intGrade != 0) {
          graded = graded :+ (films(position)._1, intGrade)
          step += 1
          position += 1
        } else {
          position += 1
        }

        if (position == films.length) {
          step = limit + 1
          println("\nNo more movies to grade\n")
        }

        if (step == limit) {

          println("Grade 20 more? y/[n]")
          var ans = scala.io.StdIn.readLine()

          if (ans == "y" || ans == "yes" || ans == "Yes" || ans == "YES") {
            limit += 20
          } else {
            println("\nFinished Grading\n")
          }

        }

      } catch {

        case e: NumberFormatException => println("Try again")
        case e: Exception => println("Unknown Error")

      }

      dumpRatings()

    }
  }

  def printRatings(): Unit = {
    val id2title = this.films.toMap
    this.graded.foreach(x => println(s"${id2title(x._1)}: ${x._2}"))
  }

  def dumpRatings() = {
    val ratings_file = "user_ratings.tsv"
    println(s"Saving ratings to ${ratings_file}\n")
    val writer = new PrintWriter(new File(ratings_file))
    this.graded.foreach(x => writer.write(s"${x._1}\t${x._2}\n"))
    writer.close()
  }

  def toRDD = {
    sc.parallelize(this.graded.map { x => Rating(0, x._1, x._2) })
  }
}
