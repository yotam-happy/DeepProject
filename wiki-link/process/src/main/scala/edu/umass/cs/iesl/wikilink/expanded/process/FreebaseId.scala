package edu.umass.cs.iesl.wikilink.expanded.process

import edu.umass.cs.iesl.wikilink.expanded.data.WikiLinkItem
import org.apache.commons.lang.StringUtils
import scala.Some
import java.io.File
import redis.clients.jedis.Jedis
import org.jsoup.nodes.Element
import org.jsoup.Jsoup
import scala.collection.JavaConverters._

/**
 * @author harshal
 * @date: 3/25/13
 */
object FreebaseId {
  val redirectsMap =  new Jedis("localhost",6380)
  val title2id = new Jedis("localhost",6381)
  val wiki2fb = new Jedis("localhost",6382)
  val Matcher = """.+/(.*)""".r

  def processThrift(file:File):WikiLinkItem ={
    val (stream, proto) = ThriftSerializerFactory.getReader(file)
    val wli = WikiLinkItem.decode(proto)
    stream.close()
    wli
  }

  def addFreebaseId(wli:WikiLinkItem):WikiLinkItem={
    val newMentions = wli.`mentions`.map(mention=>{
      mention.freebaseId match{
        case None =>{
          val wikiUrl = StringUtils.stripEnd(mention.wikiUrl,"/")
          //extract data page title from the url
          val title = wikiUrl match {
            case Matcher(temp) => temp
            case _ => """(?<=http://en.wikipedia.org/wiki/)(.*)(?!/)""".r.findFirstIn(wikiUrl).get
          }
          //get the wikipedia id for the title
          val fbId = getFbId(title)
          mention.copy(`freebaseId`= fbId)
        }
        case _ => mention
      }})
    wli.copy(`mentions`=newMentions)
  }

  def addContextAndFreebaseId(wli:WikiLinkItem):WikiLinkItem={
    lazy val anchorMap = {
      wli.content.dom match {
        case Some(dom) => {
          val doc = Jsoup.parse(dom)
          //Get all the anchor elements from the dom to compare with wikiurl
          val temp = doc.getElementsByTag("a").iterator().asScala.toSeq.filter(a=> (a.attr("href").indexOf("wikipedia.org")!= -1)).groupBy(g=>g.text().trim)
          collection.mutable.HashMap(temp.toSeq:_*)
        }
        case None => collection.mutable.HashMap[String,Seq[Element]]()
      }
    }
    val newMentions = wli.`mentions`.map(mention=>{
      (mention.context,mention.freebaseId) match{
        case (Some(context),None) => {
          val wikiUrl = StringUtils.stripEnd(mention.wikiUrl,"/")
          //extract data page title from the url
          val title = wikiUrl match {
            case Matcher(temp) => temp
            case _ => """(?<=http://en.wikipedia.org/wiki/)(.*)(?!/)""".r.findFirstIn(wikiUrl).get
          }
          //get the wikipedia id for the title
          val fbId = getFbId(title)
          mention.copy(`freebaseId`= fbId)
        }
        case (None,Some(freebaseId)) => {
          val anchors = anchorMap.get(mention.`anchorText`.trim)
          anchors match {
            case Some(as) =>{
              if(as.size>1) {
                anchorMap.update(mention.`anchorText`.trim,as.tail)
              }
              val context = Context.getContext(as.head,as.head.parent())(as.size)
              if ((context.`left`+context.`right`).length<1) {
                println("Still no context found :"+wli.`docId`)
                mention
              }
              else{
                println("Found context")
                mention.copy(`context`=Some(context))
              }
            }
            case None => {
              println("\nAnchor text not found :"+wli.`docId`)
              mention
            }
          }
        }
        case (None,None) =>{
          val wikiUrl = StringUtils.stripEnd(mention.wikiUrl,"/")
          //extract data page title from the url
          val title = wikiUrl match {
            case Matcher(temp) => temp
            case _ => """(?<=http://en.wikipedia.org/wiki/)(.*)(?!/)""".r.findFirstIn(wikiUrl).get
          }
          //get the wikipedia id for the title
          val fbId = getFbId(title)
          val anchors = anchorMap.get(mention.`anchorText`.trim)
          anchors match {
            case Some(as) =>{
              if(as.size>1) {
                anchorMap.update(mention.`anchorText`.trim,as.tail)
              }
              val context = Context.getContext(as.head,as.head.parent())(as.size)
              if ((context.`left`+context.`right`).length<1) {
                println("Still no context found :"+wli.`docId`)
                mention.copy(`freebaseId`= fbId)
              }
              else{
                println("Found context")
                mention.copy(`context`=Some(context),`freebaseId`=fbId)
              }
            }
            case None => {
              println("\nAnchor text not found :"+wli.`docId`)
              mention.copy(`freebaseId`= fbId)
            }
          }
        }
        case _ => mention
      }})
    wli.copy(`mentions`=newMentions)
  }

  def addContext(wli:WikiLinkItem):(WikiLinkItem,Boolean)={
    var flag = false
    lazy val anchorMap = {
      wli.content.dom match {
        case Some(dom) => {
          val doc = Jsoup.parse(dom)
          //Get all the anchor elements from the dom to compare with wikiurl
          val temp = doc.getElementsByTag("a").iterator().asScala.toSeq.filter(a=> (a.attr("href").indexOf("wikipedia.org")!= -1)).groupBy(g=>g.text().trim)
          collection.mutable.HashMap(temp.toSeq:_*)
        }
        case None => collection.mutable.HashMap[String,Seq[Element]]()
      }
    }
    val newMentions = wli.`mentions`.map(mention=>{
      mention.context match{
        case None =>{
          val anchors = anchorMap.get(mention.`anchorText`.trim)
          anchors match {
            case Some(as) =>{
              if(as.size>1) {
                anchorMap.update(mention.`anchorText`.trim,as.tail)
              }
              val context = Context.getContext(as.head,as.head.parent())(as.size)
              if ((context.`left`+context.`right`).length<1) {
                println("Still no context found :"+wli.`docId`)
                mention
              }
              else{
                flag=true
                println("Found context")
                mention.copy(`context`=Some(context))
              }
            }
            case None => {
              println("\nAnchor text not found :"+wli.`docId`)
              mention
            }
          }
        }
        case _ => mention
      }})
    (wli.copy(`mentions`=newMentions),flag)
  }



  def getFbId(title:String):Option[String]={
    val idOp = title2id.get(title.replaceAll("_"," ").trim.toLowerCase)
    idOp match {
      case null => None
      case id => {
        val redirect = redirectsMap.get(id)
        redirect match{
          case null => {
            wiki2fb.get(id) match {
              case null => None
              case fbId => Some(fbId)
            }
          }
          case redirect => {
            wiki2fb.get(redirect) match {
              case null => None
              case fbId => Some(fbId)
            }
          }
        }
        if (redirect==null){
          val fbId = wiki2fb.get(id)
          if(fbId==null) None
          else Some(fbId)
        }
        else{
          val fbId = wiki2fb.get(redirect)
          if(fbId==null) None
          else Some(fbId)
        }
      }
    }
  }


  def getOutDirectory(f:File):File={
    val outdir = "/iesl/local/harshal/thrifts_4_2013/thrift/"
    new File(outdir+f.getAbsolutePath.split("/")(6)+"/"+f.getName)
  }

  def main(args:Array[String]){
    var count = 0
    assert(args.length==1,"Not enough arguments")
    println("Processing dir : "+args(0))
    val dir = new File(args(0))
    //val dir = "/iesl/local/martin/thrifts_3_2013/thrift"
    //val dirs = new java.io.File(dir).listFiles().filter(_.isDirectory)
    //val chunks = dirs.grouped(dirs.size/10)
    val fList = dir.listFiles().filter(_.getName.endsWith("thrift.gz"))
    fList.foreach( f=> {
      try{
        val wli = processThrift(f)
        val outFile = getOutDirectory(f)
        if(!outFile.getParentFile.exists()) outFile.getParentFile.mkdirs()
        val (outStream, outProto) = ThriftSerializerFactory.getWriter(outFile)
        val newWli = addContextAndFreebaseId(wli)
        newWli.write(outProto)
        outStream.flush()
        outStream.close()
      }catch{
        case e:Exception => {
          println("\nException while processing "+f.getName+" : "+e.getMessage)
          println(e.getStackTraceString)
        }
      }
      count+=1
      if (count%1000==0) print(count+" ")
    })
  }
}

object AppTest extends App{
  //  val f = new File(args(0))
  val f = new File("/Users/harshal/Work/junk/001012027.thrift.gz")
  val wli = FreebaseId.processThrift(f)
//  val wli2= FreebaseId.addContext(wli)
//  println(wli2.`mentions`)
}
//find /dump -type f -name '*.xml' | parallel -j8 java -jar ProcessFile.jar {}