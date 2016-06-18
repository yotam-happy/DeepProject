package edu.umass.cs.iesl.wikilink.expanded.process

import edu.umass.cs.iesl.wikilink.expanded.data.WikiLinkItem
import java.io.File

object Viewer {
  
  def apply(f: File): String = {
    val (stream, proto) = ThriftSerializerFactory.getReader(f)
    val wli = WikiLinkItem.decode(proto)
    stream.close()
    
    val sb = new StringBuffer
    
    sb.append("docID: " + wli.docId + "\n")
    sb.append("url: " + wli.url + "\n")
    sb.append("mention.head: " + wli.mentions.head + "\n")
    sb.append("rareWords.head: " + wli.rareWords.head + "\n")
    
    val rawLength = wli.content.raw match {
      case Some(raw) => raw.limit()
      case None => -1
    }
    sb.append("raw length: " + rawLength + "\n")
    
    sb.toString
    
  }

}