# [F21] Introduction to Big Data Course. Assignment

## Authors <a name="authors"></a>
[Innopolis University](https://innopolis.university/en/) students, Data Science track.

Team **Scaladoop**:
* Daniil [@igudesman](https://github.com/igudesman) Igudesman, d.igudesman@innopolis.university, programmer
* Mikhail [@Glemhel](https://github.com/Glemhel) Rudakov, m.rudakov@innopolis.university, programmer
* Anna [@asleepann](https://github.com/asleepann) Startseva, a.startseva@innopolis.university, report writer


## Table of Contents
* [Introduction](#introduction)
* [System description](#system-description)
* [Private Network Hadoop Cluster Configuration](#hadoop-cluster)
* [Conclusion](#conclusion)

## Introduction <a name="introduction"></a>
Our team, Scaladoop, is going to implement a movie recommendation system that can help you to choose movie to watch!

To practice not only machine learning with Scala, but also the use of distributed systems, we will run our model distributed on 2 different machines using Hadoop Cluster and Spark framework for Scala to speed up computations.

This report covers all the steps that we performed and that lead us to a successful achievement of the goal.

## System description <a name="system-description"></a>

```scala
val ranks = Array(2, 5, 10, 20, 30, 40, 50, 60)
    var best_params = Array(10, Double.PositiveInfinity)
    for( r <- ranks ) {
      println(s"Rank: ${r}")
      val model = ALS.train(train.union(myRating), r, 10)

      val prediction = model.predict(test.map(x => (x.user, x.product)))

      // calculate validation error
      val recommenderRmse = rmse(test, prediction)
      println(s"Error after training: ${recommenderRmse}")

      // if baseline is implemented, calculate baseline error
      if (baseline.nonEmpty) {
        val baselineRmse = rmse(test, baseline)
        println(s"Baseline error: ${baselineRmse}")
      }

      if (recommenderRmse < best_params(1)) {
        best_params = Array(r, recommenderRmse)
      }
      println("\n")
    }
```

```scala
println("Predictions for user with filtering\n")
      val already_watched = myRating.map(x => x.product).collect()
      // for input data format refer to documentation
      // get movie ids to rank from baseline map
      model.predict(sc.parallelize(baseline.keys.map(x => (0, x)).toSeq))
        // sort by ratings in descending order
        .sortBy(_.rating, ascending = false)
        // filter non-watched movies
        .filter(x => !already_watched.contains(x.product))
        // take first 20 items
        .take(20)
        // print title and predicted rating
        .foreach(x => println(s"${filmId2Title(x.product)}\t\t${x.rating}"))
```

```scala
if (load_movie_preferences) {
    graded = sc.textFile(path + "/user_ratings.tsv")
      .map(line => line.split("\t"))
      .map(field => (field(0).toInt, field(1).toDouble))
      .collect()
      .toSeq
```

## Private Network Hadoop Cluster Configuration <a name="hadoop-cluster"></a>
Steps of configuring local network for running our movie recommendation system on 2 computers both connnected to the same Wi-Fi:
1. Configuring hostnames:
<img src="https://github.com/asleepann/IBD-Assignment/blob/main/images-for-report/local_net.png"/> 
2. Add "hadoop" user to the system using the following commands:

```bash
sudo useradd -m hadoop
sudo passwd hadoop
sudo adduser hadoop sudo
```
<img src="https://github.com/asleepann/IBD-Assignment/blob/main/images-for-report/add_hadoop_user.png"/>
3. d
4. 

## Conclusion <a name="conclusion"></a>
Our team successfully implemented the movie reccomendation system and run it on 2 different computers.
