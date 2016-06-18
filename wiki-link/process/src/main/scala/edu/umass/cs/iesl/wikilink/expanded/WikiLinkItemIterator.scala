package edu.umass.cs.iesl.wikilink.expanded

//import edu.umass.cs.iesl.wikilink.expanded.WikiLinkItemIterator.MyJsonProtocol._
//import edu.umass.cs.iesl.wikilink.expanded.data.MyJsonProtocol._

import java.util.NoSuchElementException

import edu.umass.cs.iesl.wikilink.expanded.data.WikiLinkItem
import java.io.File
import org.apache.thrift.transport.TTransportException
import edu.umass.cs.iesl.wikilink.expanded.process.ThriftSerializerFactory
import spray.json._
import java.io.{BufferedWriter,FileWriter}

import scala.collection.mutable.ListBuffer

case class wikiJsonObj(word: String, left_context: String, right_context: String, wikiurl: String, url: String, freebaseID: String, docId: Int)

object MyJsonProtocol extends DefaultJsonProtocol{
  implicit val wikiFormat = jsonFormat7(wikiJsonObj.apply)
}

import MyJsonProtocol._
import spray.json._

object WikiLinkItemIterator {

  def recursiveFileIterator(d: File, fileFilter: (File) => Boolean = f => true): Iterator[File] = {
    val these = d.listFiles
    these.filter(fileFilter).iterator ++ these.filter(_.isDirectory).iterator.flatMap(d => recursiveFileIterator(d, fileFilter))
  }

  def getFiles(d: File): Iterator[File] = {
    assert(d.isDirectory, d.getAbsolutePath + " is not a directory.")
    recursiveFileIterator(d, f => f.isFile && f.getName.endsWith(".gz"))
  }

  def apply(dirName: String): Iterator[WikiLinkItem] = apply(new File(dirName))

  def apply(d: File): Iterator[WikiLinkItem] = getFiles(d).flatMap(f => new PerFileWebpageIterator(f))

  def PharseToJson(wli: WikiLinkItem, w: BufferedWriter, flag: Boolean): ListBuffer[Int] = {
    var indx = wli.mentions.size
    var bad_wiki_indx = new ListBuffer[Int]()
    for(k <- 1 to indx){
      if(wli.`mentions`(k-1).context == None){
        bad_wiki_indx += k // TODO: add to array the bad references
      }
      else {
        var docId = wli.`docId`
        var context_left = wli.`mentions`(k - 1).context.get.left
        var context_right = wli.`mentions`(k - 1).context.get.right
        var word = wli.`mentions`(k - 1).anchorText
        var wiki_url = wli.`mentions`(k - 1).wikiUrl
        var url_address = wli.`url`
        var freebase_id = wli.`mentions`(k - 1).freebaseId.toString
        val wjo = wikiJsonObj(word, context_left, context_right, wiki_url, url_address, freebase_id, docId)
        w.write(wjo.toJson.prettyPrint)
        if(flag) {
         w.write(",") // TODO: Can be a problem if last mention has no context
        }
      }
    }
    return bad_wiki_indx
  }

  def main(args: Array[String]) = {
    val it = WikiLinkItemIterator("C:\\Users\\Noam\\Documents\\Data_DeepESA\\Wiki_thrift")
//    val it = WikiLinkItemIterator(args(0))
    var num_of_bad_links = 0
    var c = 0
    var m = 0
    var reset_counter_all = 0
    var reset_counter_bad = 0
    var file_ind = 1
    var file_ind_bad = 1
    var opener : String = """{"wlinks":["""
    var closer : String = "]}"
    var w = new BufferedWriter(new FileWriter("wikilink_0.json"))
    var logFile = new BufferedWriter(new FileWriter("log_0.json"))
    w.write(opener)
    w.write("\n")
    logFile.write(opener)
    logFile.write("\n")
    for (wli <- it) {
      if((reset_counter_all - reset_counter_bad) > 500000){
        w.write("\n")
        w.write(closer)
        w.close
        w = new BufferedWriter(new FileWriter("wikilink_"+file_ind+".json"))
        w.write(opener)
        w.write("\n")
        reset_counter_all = 0
        file_ind += 1
      }
      if(reset_counter_bad > 500000){
        w.write("\n")
        logFile.write(closer)
        logFile.close
        logFile = new BufferedWriter(new FileWriter("log_"+file_ind_bad+".json"))
        logFile.write(opener)
        w.write("\n")
        reset_counter_bad = 0
        file_ind_bad += 1
      }

      var bad_indx = PharseToJson(wli, w, it.hasNext)

      if(!bad_indx.isEmpty){
        var itr = bad_indx.iterator
        while(itr.hasNext){
          var indx = itr.next()
          var docId = wli.`docId`
          var url_address = wli.`url`
          var wiki_url = wli.`mentions`(indx-1).wikiUrl
          var word = wli.`mentions`(indx-1).anchorText
          var freebase_id = wli.`mentions`(indx-1).freebaseId.toString
          val log_wiki = wikiJsonObj(word,"None","None",wiki_url,url_address,freebase_id,docId)
          logFile.write(log_wiki.toJson.prettyPrint)
          if(it.hasNext) {
            logFile.write(",")
          }
        }
        num_of_bad_links += bad_indx.length
        reset_counter_bad += bad_indx.length
      }
        m += wli.mentions.size
        reset_counter_all += wli.mentions.size
        c += 1
    }
    w.write(closer)
    w.close
    logFile.write(closer)
    logFile.close
    println("Total Pages : " + c)
    println("Total Mentions : " + m)
    println("Total bad mentions : " + num_of_bad_links)
  }
}

class PerFileWebpageIterator(f: File) extends Iterator[WikiLinkItem] {

  var done = false
  val (stream, proto) = ThriftSerializerFactory.getReader(f)
  private var _next: Option[WikiLinkItem] = getNext()

  private def getNext(): Option[WikiLinkItem] = try {
    Some(WikiLinkItem.decode(proto))
  } catch {case _: TTransportException => {done = true; stream.close(); None}}

  def hasNext(): Boolean = !done && (_next != None || {_next = getNext(); _next != None})

  def next(): WikiLinkItem = if (hasNext()) _next match {
    case Some(wli) => {_next = None; wli}
    case None => {throw new Exception("Next on empty iterator.")}
  } else throw new Exception("Next on empty iterator.")

}