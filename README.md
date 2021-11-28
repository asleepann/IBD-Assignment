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
* [Results Analysis](#results-analysis)
* [Conclusion](#conclusion)

## Introduction <a name="introduction"></a>
Our team, Scaladoop, is going to implement a movie recommendation system that can help you to choose movie to watch!

To practice not only machine learning with Scala, but also the use of distributed systems, we will run our model distributed on 2 different machines using Hadoop Cluster and Spark framework for Scala to speed up computations.

This report covers all the steps that we performed and that lead us to a successful achievement of the goal.

## System description <a name="system-description"></a>

We were given part of the code, so we only list here important changes we applied.

In general, the datasets with users, watched film and their rating is provided. System is to learn on this data and to predict rating for other users and films.

We also let the user choose: if load_movie_preferences is true, they are automatically read from user_ratings.tsv file. Otherwise, user is prompted to rate movies at the start of the program.
```scala
if (load_movie_preferences) {
    graded = sc.textFile(path + "/user_ratings.tsv")
      .map(line => line.split("\t"))
      .map(field => (field(0).toInt, field(1).toDouble))
      .collect()
      .toSeq
```


We filter out films that user already watched, so they are not shown in recommendations:
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


We also perform fine-tuning of parameters: we try different *rank* for our model to see what is the best. The best result is discussed at the end of the report.
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

3. Hadoop configuration files should contain the following:
 
* `yarn-site.xml`

```xml
<configuration>
     <property>
            <name>yarn.nodemanager.disk-health-checker.min-healthy-disks</name>
            <value>0.0</value>
     </property>
     <property>
            <name>yarn.nodemanager.disk-health-checker.max-disk-utilization-per-disk-percentage</name>
            <value>100.0</value>
    </property>
    <property>
            <name>yarn.resourcemanager.hostname</name>
            <value>mi-msi</value>
    </property>
    <property>
            <name>yarn.resourcemanager.webapp.address</name>
            <value>mi-msi:8088</value>
    </property>
    <property>
            <name>yarn.resourcemanager.address</name>
            <value>mi-msi:8032</value>
    </property>
    <property>
            <name>yarn.resourcemanager.scheduler.address</name>
            <value>mi-msi:8030</value>
    </property>
    <property>
            <name>yarn.resourcemanager.resource-tracker.address</name>
            <value>mi-msi:8031</value>
    </property>
    <property>
       <name>yarn.nodemanager.aux-services</name>
       <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.acl.enable</name>
        <value>0</value>
    </property>
    <property>
   <name>yarn.scheduler.capacity.root.support.user-limit-factor</name>  
   <value>2</value>
</property>

<property>
   <name>yarn.nodemanager.vmem-check-enabled</name>
    <value>false</value>
    <description>Whether virtual memory limits will be enforced for containers</description>
  </property>
 <property>
   <name>yarn.nodemanager.vmem-pmem-ratio</name>
    <value>4</value>
    <description>Ratio between virtual memory to physical memory when setting memory limits for containers</description>
  </property>
</configuration>
```

* `core-site.xml`

```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://mi-msi:9000</value>
    </property>
</configuration>
```

* `hdfs-site.xml`

```xml
<configuration>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/home/hadoop/hadoop_tmp</value>
    </property>

    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
</configuration>
```

* `mapred-site.xml`

```xml
<configuration>
    <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
    </property>
    <property>
            <name>yarn.app.mapreduce.am.env</name>
            <value>HADOOP_MAPRED_HOME=/home/hadoop/hadoop</value>
    </property>
    <property>
            <name>mapreduce.map.env</name>
            <value>HADOOP_MAPRED_HOME=/home/hadoop/hadoop</value>
    </property>
    <property>
           <name>mapreduce.reduce.env</name>
           <value>HADOOP_MAPRED_HOME=/home/hadoop/hadoop</value>
    </property>
</configuration>
```

* `workers`

```plain text
hadoop@mi-msi
hadoop@igudesman-2x
```

4. The result of the command  `hdfs dfsadmin -report`

<img src="https://github.com/asleepann/IBD-Assignment/blob/main/images-for-report/hdfs_dfsadmin.png"/>

HDFS is up on 2 hosts.

5. We downloaded data into HDFS

<img src="https://github.com/asleepann/IBD-Assignment/blob/main/images-for-report/hdfs_download_data.png"/>

6. We ran HDFS and YARN

<img src="https://github.com/asleepann/IBD-Assignment/blob/main/images-for-report/hd.png"/>
<img src="https://github.com/asleepann/IBD-Assignment/blob/main/images-for-report/ig.jpg"/>

7. We started execution and got the following results for both modes (grader and non-grader)

<img src="https://github.com/asleepann/IBD-Assignment/blob/main/images-for-report/res.png"/>
<img src="https://github.com/asleepann/IBD-Assignment/blob/main/images-for-report/res2.png"/>

## Results Analysis <a name="results-analysis"></a>
We tried different values of rank in the model:

<img src="https://github.com/asleepann/IBD-Assignment/blob/main/images-for-report/graph_rank_error.png"/>

According to graph, models with rank higher 10 are overfitting. Model with rank 10 produces lowest error after training, therefore, 10 is the optimal value for rank.

## Conclusion <a name="conclusion"></a>
Our team successfully implemented the movie reccomendation system and run it on 2 different computers.
