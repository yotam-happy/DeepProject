package edu.umass.cs.iesl.wikilink.google.analysis

import org.sameersingh.utils.cmdopts.CmdLine
import org.sameersingh.utils.timing.TimeUtil
import edu.umass.cs.iesl.wikilink.google.WebpageIterator

/**
 * @author sameer
 * @date 10/15/12
 */
object WikiURLDistribution {
  def main(args: Array[String]) {
    val opts = CmdLine.parse(args)
    println(opts)
    val dirName = opts.get("dir")
    val output = opts.getOrElse("output", "output")

    if (dirName.isEmpty) {
      println("Usage: mvn scala:run -DmainClass=edu.umass.cs.iesl.wikilink.google.analysis.WikiURLDistribution -DaddArgs=\"@dir=/dir/containing/google/gz/files\"")
      sys.exit(1)
    }
    TimeUtil.init
    var mentionId: Long = 0
    val wikiMap: Clustering = new WikiURLMap
    for (page <- WebpageIterator(dirName.get)) {
      if (page.id % 100000 == 0) {
        TimeUtil.snapshot("Done %d".format(page.id))
      }
      for (mention <- page.mentions) {
        mentionId += 1
        wikiMap.addMention(mention, mentionId)
      }
    }
    TimeUtil.snapshot(wikiMap.toString)
    wikiMap.writeSizesToFile(output)
  }
}
