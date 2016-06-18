package edu.umass.cs.iesl.wikilink.google

import collection.mutable.ArrayBuffer
import java.io.BufferedReader
import java.util.regex.Pattern

/**
 * @author sameer
 * @date 3/8/12
 */

case class RareWord(val word: String, val offset: Int)

object RareWord {
  def fromText(rareWordStr: String): RareWord = {
    val splits = rareWordStr.split("\\t")
    assert(splits.length == 3, rareWordStr)
    assert(splits(0) == Webpage.TOKEN, rareWordStr)
    val text = splits(1)
    val offset = splits(2).toInt
    new RareWord(text, offset)
  }
}

case class Mention(val text: String, val offset: Int, val url: String)

object Mention {

  val matcher = Pattern.compile(",[1-9][0-9]+,(s?http|ftp|file)").matcher("")

  def fromText(mentionStr: String): Mention = {
    val splits = mentionStr.split("\\t")
    assert(splits.length == 4, mentionStr)
    assert(splits(0) == Webpage.MENTION, mentionStr)
    val text = splits(1)
    val offset = splits(2).toInt
    val wikiURL = splits(3)
    new Mention(text, offset, wikiURL)
  }
}

case class Webpage(val id: Int,  val url: String, val mentions: Seq[Mention], val rareWords: Seq[RareWord]) {
  override def toString = "url:\t%s\nmentions:\n%s\nwords:\n%s".format(url, mentions.mkString("\t", "\n\t", ""), rareWords.mkString("\t", "\n\t", ""))
}

object Webpage {
  val URL = "URL"
  val MENTION = "MENTION"
  val TOKEN = "TOKEN"

  def getNext(stream: BufferedReader, id: Int = -1): Webpage = {
    val urlLine = stream.readLine.trim()
    val urlSplits = urlLine.split("\\t")
    assert(urlSplits.length == 2, urlLine)
    assert(urlSplits(0).equals(URL), urlLine)
    val url = urlSplits(1).trim
    val mentionLines = new ArrayBuffer[String]
    val wordLines = new ArrayBuffer[String]
    var line = stream.readLine().trim
    while(line != "") {
      val splits = line.split("\\t")
      splits.head match {
        case MENTION => mentionLines += line
        case TOKEN   => wordLines += line
      }
      line = stream.readLine()
    }

    // another line is empty
    line = stream.readLine()
    assert(line.trim.size == 0, line)

    // create the webpage
    val mentions = mentionLines.map(Mention.fromText(_))
    val rare = wordLines.map(RareWord.fromText(_))
    new Webpage(id, url.trim, mentions, rare)
  }
}