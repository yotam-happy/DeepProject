package edu.umass.cs.iesl.wikilink.google

import java.io._
import java.util.zip.GZIPInputStream
import org.sameersingh.utils.cmdopts.CmdLine

/**
 * @author sameer
 * @date 3/8/12
 */

class PerFileWebpageIterator(val filename: String) extends Iterator[Webpage] {
  val reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(filename))))
  var count = -1

  def hasNext = {
    val bool = reader.ready
    if (!bool) reader.close
    bool
  }

  def next() = {
    // println(count)
    count += 1
    Webpage.getNext(reader, count)
  }
}

object MentionIterator {
  def apply(filenames: Seq[String]): Iterator[Mention] = WebpageIterator(filenames).flatMap(_.mentions)

  def apply(dirName: String): Iterator[Mention] = apply(WebpageIterator.getFiles(dirName))
}

object WebpageIterator {
  def getFiles(dirName: String): Seq[String] = {
    val dir = new File(dirName)
    assert(dir.exists)
    assert(dir.isDirectory)
    val filenames = dir.listFiles(new FilenameFilter {
      def accept(p1: File, p2: String) = p2.endsWith(".gz")
    }).map(_.getAbsolutePath).sorted
    // println(filenames.mkString(", ")
    filenames.toSeq
  }

  def apply(filenames: Seq[String]): Iterator[Webpage] =
    filenames.iterator.map(f => new PerFileWebpageIterator(f)).flatMap(_.toIterator)

  def apply(dirName: String): Iterator[Webpage] = apply(getFiles(dirName))
}

object WebpageIteratorRunner {
  def main(args: Array[String]) {
    val opts = CmdLine.parse(args)
    println(opts)
    val dirName = opts.get("dir")
    val takeOnly = opts.getOrElse("take", Int.MaxValue.toString).toInt

    if (dirName.isEmpty) {
      println("Usage: mvn scala:run -DmainClass=edu.umass.cs.iesl.wikilink.google.analysis.WebpageIteratorRunner" +
            " -DaddArgs=\"@dir=/dir/containing/google/gz/files\"")
      sys.exit(1)
    }
    val iterator = WebpageIterator(dirName.get).take(takeOnly)
    iterator.foreach(w => println(w))
  }
}