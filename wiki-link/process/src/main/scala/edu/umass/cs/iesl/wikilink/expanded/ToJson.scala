package edu.umass.cs.iesl.wikilink.expanded

import edu.umass.cs.iesl.wikilink.expanded.data.WikiLinkItem

/**
 * Created by Noam on 6/13/2016.
 */
class ToJson {
  def main(args: Array[String]) {
    val path: Array[String] = new Array[String](1)
    path(1) = "C:\\Users\\Noam\\OneDrive for Business\\NLP for DL 2016\\Project\\data\\wikilinks_data"
    val it = WikiLinkItemIterator("C:\\Users\\Noam\\OneDrive for Business\\NLP for DL 2016\\Project\\data\\wikilinks_data")
    println("success!")
    return
  }
}
