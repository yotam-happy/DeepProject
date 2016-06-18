package edu.umass.cs.iesl.wikilink.expanded.process
import scala.xml.PrettyPrinter
import scala.xml.XML
import scala.io.Source
import org.ccil.cowan.tagsoup.jaxp.SAXFactoryImpl
import java.io.BufferedReader
import java.io.StringReader

object CleanDOM {
  
  def apply(rawHTML: String): Option[String] = {
    try {
	    
	  // instantiate the parser
	  val parser = XML.withSAXParser((new SAXFactoryImpl).newSAXParser())
	  val pp = new PrettyPrinter(300, 2)
	    
	  // parse the file
	  val htmlReader = new BufferedReader(new StringReader(rawHTML))
	  val html = parser.load(htmlReader)
	    
	  // pretty-print the DOM
	  val sb = new StringBuilder
	  pp.format(html, sb)
	  Some(sb.toString())
	    
    } catch {
	   case _ => None
	}
  }

}