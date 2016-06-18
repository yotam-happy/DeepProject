package edu.umass.cs.iesl.wikilink.expanded.process

import org.jsoup.nodes.Element
import edu.umass.cs.iesl.wikilink.expanded.data.{Context => ContextWrapper, WikiLinkItem, Mention}
import java.util.regex.Pattern
import org.jsoup.Jsoup
import collection.mutable
import collection.mutable.{HashMap, ArrayBuffer}
import org.apache.commons.lang.StringUtils
import io.Source
import scala.collection.JavaConverters._
import java.io.{InputStreamReader, BufferedReader}
import java.util.zip.GZIPInputStream

/**
 * @author harshal
 * @date: 2/27/13
 */
object Context {
  val window_size = 25
  lazy val redirectsMap =  Loaders.loadRedirectsMap("redirects.tsv.gz")
  lazy val title2id = Loaders.loadTitleIdMap("wpid-title-mapping.tsv.gz")
  lazy val wiki2fb = Loaders.loadWikiToFreebaseMap("freebase-wex-2012-08-06-freebase_wpid.tsv.gz")

  /**
   * takes a wikilinkitem and returns a seq[Mention] with context and freebaseid filled in
   * @param wli
   * @return
   */

  def addContextAndFreebaseId(wli:WikiLinkItem):WikiLinkItem={
    def newMentionArray = ArrayBuffer[Mention]()

    val _map = mutable.HashMap[(String,String),(ArrayBuffer[Mention],Int)]()
    wli.mentions.foreach(mention=>{
      val res = _map.getOrElseUpdate((pruneUrl(mention._1),mention.`anchorText`),(newMentionArray,0))
      _map((pruneUrl(mention._1),mention.`anchorText`)) = Tuple2(res._1+=mention,res._2+1)
    })

    wli.content.dom match {
      case Some(dom) => {
        val mentions = processDom(dom,_map,wli.docId)
        wli.copy(`mentions`= mentions)
      }
      case None => wli
    }
  }

  /**
   * Recursively find a large enough context for the anchor by looking at the parents
   * Implicit int i is used when dealing with situations where multiple multiple mention
   * exist for the same anchor tag.
   * @param anchor
   * @param parent
   * @return  Context
   */
  @annotation.tailrec
  def getContext(anchor:Element,parent:Element)(implicit i:Int):ContextWrapper = {
    //Split the actual html content for this element by the anchor's actual html
    val part = parent.html().split(Pattern.quote(anchor.toString))
    val numParts = i+1
    val (lind,rind)=if (part.length>=numParts) (i-1,i) else (0,1)
    val left = if (part.length>0) Jsoup.parse(part(lind)).text() else ""
    val right = if (part.length>1) Jsoup.parse(part(rind)).text() else ""
    if ((left.split("""\s+""").length>=window_size && right.split("""\s+""").length>=window_size) || parent.parent() == null){
      var split = left.split("""\s+""")
      val leftStr = if (split.length<=window_size) left else split.takeRight(window_size).mkString(" ")
      split = right.split("""\s+""")
      val rightStr = if (split.length<=window_size) right else split.take(window_size).mkString(" ")
      return ContextWrapper(leftStr,rightStr,anchor.text())
    }
    else{
      getContext(anchor,parent.parent())
    }
  }

  def processDom(html:String,_map:mutable.HashMap[(String,String),(ArrayBuffer[Mention],Int)],id:Int)
  :Seq[Mention] = {
    val Matcher = """.+/(.*)""".r
    val doc = Jsoup.parse(html)
    //Get all the anchor elements from the dom to compare with wikiurl
    val anchors = doc.getElementsByTag("a").iterator()

    //Do this only if context was not found for some mentions in the page
    lazy val anchorMap = {
      val temp = anchors.asScala.toSeq.filter(a=> (a.attr("href").indexOf("wikipedia.org")!= -1)).groupBy(g=>g.text().trim)
      collection.mutable.HashMap(temp.toSeq:_*)
    }

    val mentions = ArrayBuffer[Mention]()
    while(!_map.isEmpty && anchors.hasNext){
      val anchor = anchors.next()
      val anchorText = anchor.text().trim
      val href = anchor.attr("href")
      val prunedHref = pruneUrl(href)
      if(!prunedHref.equals("")){ //Check if its a wikipedia url
        if (_map.contains((prunedHref,anchorText))){
          val value = _map.get((prunedHref,anchorText)).get
          /**
           * there could be multiple links for the same url in the page
           * this is stored as a list in the corresponding map bucket
           * hence when a url is matched it should be removed list so it's not matched twice
           * when the array is empty remove corresponding map entry
           */
          val mention = value._1.remove(0)
          if(value._1.length==0) _map.remove((prunedHref,anchorText))
          else _map((prunedHref,anchorText)) = value
          //Implicit value to be passed to getContext
          implicit val index_within_map_array = value._2-value._1.length

          val fbId = getFbId(mention.wikiUrl)

          //generate a mention with context and fb id filed in
          mentions+= mention.copy(`context`=Some(getContext(anchor,anchor.parent())),
            `freebaseId`=fbId)
        }
      }
    }
    //add the mentions not found in page back with empty context and freebase id
    mentions++=_map.values.map(m=>{
      m._1.map(mention=>{
        //add freebase id
        val fbId = getFbId(mention.wikiUrl)
        val anchors = anchorMap.get(mention.`anchorText`.trim)
        val contextOp = anchors match {
          case Some(as) =>{
            if(as.size>1) {
              anchorMap.update(mention.`anchorText`.trim,as.tail)
            }
            Some(Context.getContext(as.head,as.head.parent())(as.size))
          }
          case None => None
        }

        mention.copy(`context`=contextOp,`freebaseId`=fbId)
      })
    }).flatten
  }

  def getFbId(url:String):Option[String]={
    val Matcher = """.+/(.*)""".r
    val wikiUrl = StringUtils.stripEnd(url,"/")
    //extract data page title from the url
    val title = wikiUrl match {
      case Matcher(temp) => temp
      case _ => """(?<=http://en.wikipedia.org/wiki/)(.*)(?!/)""".r.findFirstIn(wikiUrl).get
    }

    //get the wikipedia id for the title
    val idOp = title2id.get(title.replaceAll("_"," ").trim.toLowerCase)
    val fbId = idOp match {
      case Some(id) => wiki2fb.get(redirectsMap.getOrElse(id,id))
      case None => None
    }
    fbId
  }


  // pruning is necessary because sites might garbled urls like: <a href="//en.wikipedia...."> (that is, no "http:")
  def pruneUrl(url: String): String = {
    val index = url.indexOf("wikipedia.org")
    if(index != -1) url.substring(index) else ""
  }


  /**
   * Test case, uncomment and run
   */
  //  def main(args:Array[String]){
  //    val wli = edu.umass.cs.iesl.data.WikiLinkItem(
  //      docId = 1,
  //      url = "",
  //      content = PageContentItem(
  //        raw = null,
  //        fullText = null,
  //        articleText = null,
  //        dom = Some("""<!DOCTYPE html><html><head prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# githubog: http://ogp.me/ns/fb/githubog#"><title></title></head>"""+
  //                  """<body class="logged_in page-profile macintosh  env-production  "><div id="wrapper"><p><a href="http://en.wikipedia.org/data/Augustin-Louis_Cauchy">Baron Augustin-Louis Cauchy</a> was a French mathematician who was an early pioneer of analysis. He started the project of formulating and proving the theorems of infinitesimal calculus in a rigorous manner, rejecting the heuristic principle of the generality of algebra exploited by earlier authors.</p>"""+
  //                  """<p>"More concepts and theorems have been named for Cauchy than for any other mathematician (in elasticity alone there are sixteen concepts and theorems named for <a href="http://en.wikipedia.org/data/Augustin-Louis_Cauchy">Cauchy</a>)."</p>"""+
  //                    """<p>Cauchy was a prolific writer; he wrote approximately eight hundred research articles and five complete textbooks. He was a devout Roman Catholic, strict Bourbon royalist, and a close associate of the Jesuit order.</p>"""+
  //                    """<p><a href="http://en.wikipedia.org/data/Augustin-Louis_Cauchy"> Cauchy </a> was the son of Louis François Cauchy (1760–1848) and Marie-Madeleine Desestre. Cauchy had two brothers, Alexandre Laurent <a href="http://en.wikipedia.org/data/Augustin-Louis_Cauchy"> Cauchy </a> (1792–1857), who became a president of a division of the court of appeal in 1847, and a judge of the court of cassation in 1849; and Eugene François Cauchy (1802–1877), a publicist who also wrote several mathematical works."""+
  //                """Cauchy married Aloise de Bure in 1818. She was a close relative of the publisher who published most of Cauchy's works. By her he had two daughters, Marie Françoise Alicia (1819) and Marie Mathilde (1823).</p>"""+
  //                    """</div></body></html>""")
  //      ),
  //      rareWords = null,
  //      mentions = Seq(Mention(
  //      "http://en.wikipedia.org/data/Augustin-Louis_Cauchy",
  //      "Baron Augustin-Louis Cauchy",
  //      0),
  //        Mention(
  //          "http://en.wikipedia.org/data/Augustin-Louis_Cauchy",
  //          "Cauchy",
  //          0),
  //        Mention(
  //          "http://en.wikipedia.org/data/Augustin-Louis_Cauchy",
  //          "Cauchy",
  //          0),
  //        Mention(
  //          "http://en.wikipedia.org/data/Augustin-Louis_Cauchy",
  //          "Cauchy",
  //          0))
  //    )
  //    addContextAndFreebaseId(wli).`mentions`.foreach(println(_))
  //
  //  }
}

object Loaders {
  def loadWikiToFreebaseMap(wikiToFreeBaseFile:String) : HashMap[String,String] ={
    println("Loading from "+wikiToFreeBaseFile+"...")
    val map = HashMap[String,String]()
    println("### "+getClass.getResource(wikiToFreeBaseFile).toString)
    val reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(this.getClass.getResourceAsStream(wikiToFreeBaseFile))))
    var line = reader.readLine()
    while(line!=null){
      val split = line.split("\t")
      if(split.size==2) map(split(1))=split(0)
      line = reader.readLine()
    }
    reader.close()
    map
  }

  def loadTitleIdMap(titleIdFile:String) : HashMap[String,String]={
    println("Loading from "+titleIdFile+"...")
    val map = HashMap[String,String]()
    println("### "+getClass.getResource(titleIdFile).toString)
    val reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(this.getClass.getResourceAsStream(titleIdFile)),"UTF-8"))
    var line = reader.readLine()
    while(line!=null){
      val split = line.split("\t")
      if(split.size ==2) map(split(0).toLowerCase) = split(1)
      line = reader.readLine()
    }
    reader.close()
    map
  }
  def loadRedirectsMap(file:String) : HashMap[String,String] = {
    println("Loading from "+file+"...")
    println("### "+getClass.getResource(file).toString)
    val map = HashMap[String,String]()
    val reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(this.getClass.getResourceAsStream(file))))
    var line = reader.readLine()
    while(line!=null){
      val split = line.split("\t")
      if(split.size ==2) map(split(0)) = split(1)
      line = reader.readLine()
    }
    reader.close()
    map
  }
}