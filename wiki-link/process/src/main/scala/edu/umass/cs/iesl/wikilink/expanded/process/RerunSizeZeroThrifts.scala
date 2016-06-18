package edu.umass.cs.iesl.wikilink.expanded.process

import java.io.File

object RerunSizeZeroThrifts {

  import Runner._

  def main(args: Array[String]): Unit = {

    val idsToRerun: Seq[Int] = readFileString(new File(args(0))).split("\n").map(_.toInt)
    val pagesDir = new File(args(1))
    val googleDir = new File(args(2))
    val thriftDir = new File(args(3))

    if (!googleDir.isDirectory() ||
        !thriftDir.isDirectory())
      println("Usage: ./run OWPL-id-file pages-dir google-dir thrift-dir")

    for (id <- idsToRerun) {
      try {
        val pageFile = pageFileOption(pagesDir, id)
        val googleFile = new File(googleDir.getAbsolutePath() + ("/%09d" format id))
        val thriftFile = new File(thriftDir.getAbsolutePath() + ("/%09d.thrift.gz" format id))
        thriftFile.delete()
        thriftFile.createNewFile()
        serialize(id, pageFile, googleFile, thriftFile)
      } catch {
        case e => {
          println("skipping " + id)
          println(e.getCause())
          println(e.getMessage())
          println(e.getStackTraceString)
        }
      }
    }

  }
}
