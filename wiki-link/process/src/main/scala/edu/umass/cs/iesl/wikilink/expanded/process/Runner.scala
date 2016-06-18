package edu.umass.cs.iesl.wikilink.expanded.process

import edu.umass.cs.iesl.wikilink.expanded.data._
import java.nio.ByteBuffer
import io.Source
import java.nio.charset.Charset
import java.io.File
import scala.collection.mutable.ListBuffer
import java.io.BufferedReader
import java.io.FileReader
import java.io.FileInputStream
import java.nio.channels.FileChannel

object Runner {
  
  def main(args: Array[String]): Unit = {
    
    val pagesChunkDir = new File(args(0)) 
    val googleDir = new File(args(1))
    val thriftDir = new File(args(2))
    
    if (!googleDir.isDirectory() ||
        !thriftDir.isDirectory())
      println("Usage: ./run pages-chunk-dir google-dir thrift-dir")
    
    println("Chunk dir: " + pagesChunkDir.getAbsolutePath())
    // for all the files in each chunk that don't have an extension:
    // [chunks are of size 1000]
    val chunkSize  = 1000
    val chunkStart = pagesChunkDir.getName().toInt * chunkSize + 1 
    val chunkEnd   = chunkStart + chunkSize
    
    for (i <- chunkStart until chunkEnd) {
      // get page file option [could probably be more concise]
      val pageFile = pageFileOption(pagesChunkDir, i)
      
      // make and serialize out the thrift file
      try {
        val googleFile = new File(googleDir.getAbsolutePath() + ("/%09d" format (i - 1)))
        new File(thriftDir.getAbsolutePath() + ("/%06d".format((i / chunkSize).toInt))).mkdir()
        val thriftFile = new File(thriftDir.getAbsolutePath() + ("/%06d/%09d.thrift.gz".format((i / chunkSize).toInt, i)))
        thriftFile.createNewFile()
        serialize(i, pageFile, googleFile, thriftFile)
      } catch {
        case e => {
          println("skipping " + i)
          println(e.getCause())
          println(e.getMessage())
          println(e.getStackTraceString)
        }
      }
    }
  }
    
  
  def pageFileOption(pagesChunkDir: File, i: Int): Option[File] = {
    var res: Option[File] = None
    try {
      val f = new File("%s/%d" format (pagesChunkDir.getAbsolutePath(), i))
      if (f.exists())
        res = Some(f)
      else
        println("Page file not found: %s" format f.getAbsolutePath())
    }
    catch {
      case _ => res = None
    }
    res
  }
  
  def readFileString(f: File): String = {
    readRawFile(f) match {
      case Some(bb) => {
        val str = Charset.defaultCharset().decode(bb).toString()
        bb.rewind()
        str
      }
      case None => ""
    }
  }
  
  def readRawFile(f: File): Option[ByteBuffer] = {
    var res: Option[ByteBuffer] = None
    var stream: FileInputStream = null
    try {
      val stream = new FileInputStream(f)
      val fc = stream.getChannel()
      val bArray = Array.ofDim[Byte](f.length.toInt)
      val bb = ByteBuffer.wrap(bArray)
      fc.read(bb)
      bb.rewind()
      res = Some(bb)
    }
    finally {
      if (stream ne null)
        stream.close()
    } 
    res
  }

  def serialize(id: Int, pageFile: Option[File], googleFile: File, thriftFile: File): Unit = {
    
    val rawHtml: Option[ByteBuffer] = {
      pageFile match {
        case Some(pageFile) => readRawFile(pageFile)
        case None => None
      }
    }
    
    val html = rawHtml match {
      case Some(rawHtml) => {
        val res = Charset.defaultCharset().decode(rawHtml).toString()
        rawHtml.rewind()
        res
      }
      case None => ""
    }
    
    val (gurl, gmentions, grarewords) = googleAnnotations(googleFile)
    
    var s = WikiLinkItem(
       docId = id,
       url = gurl,
       content = PageContentItem(
           rawHtml,
           FullText(html),
           ArticleText(html),
           CleanDOM(html)
       ),
       rareWords = grarewords,
       mentions = gmentions
    )

    try {
      val sWithCtx = Context.addContextAndFreebaseId(s)
      s = sWithCtx
    } catch {
      case e => println("caught context exception: " + e.getMessage + e.getStackTraceString + e.getCause)
    }

    val (outStream, outProto) = ThriftSerializerFactory.getWriter(thriftFile)
    
    s.write(outProto)
    
    outStream.flush()
    outStream.close()
    
  }
  
  val URLMatch = "URL\t(.*)".r
  val MentionMatch = "MENTION\t([^\t]*)\t([0-9]*)\t(.*)".r
  val TokenMatch = "TOKEN\t([^\t]*)\t([0-9]*)".r
  
  def googleAnnotations(googleFile: File) = {
    
    var url: String = null
    var mentions = ListBuffer[Mention]()
    var rareWords = ListBuffer[RareWord]()
    
    for (line <- readFileString(googleFile).split("\n")) {
      line match {
        case URLMatch(_url) => { url = _url }
        case MentionMatch(text, offset, url) =>  { mentions += Mention(wikiUrl = url, anchorText = text, rawTextOffset = offset.toInt) }
        case TokenMatch(word, offset) =>  { rareWords += RareWord(word, offset.toInt) }
        case _ => ()
      }
    }
    
    (url, mentions, rareWords)
  }

}
