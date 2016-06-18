package edu.umass.cs.iesl.wikilink.google.analysis

import java.net.URL
import java.io.PrintWriter
import collection.mutable.HashMap
import org.sameersingh.utils.coref.{CorefEvaluator, EntityMap}
import org.sameersingh.utils.timing.TimeUtil
import org.sameersingh.utils.cmdopts.CmdLine
import edu.umass.cs.iesl.wikilink.google.{WebpageIterator, Mention}

/**
 * @author sameer
 * @date 3/16/12
 */

abstract class Clustering extends EntityMap {

  val entityIdMap = new HashMap[String, Long]

  def getEntityString(mention: Mention): String

  def getEntityId(mention: Mention): Long = {
    val string = getEntityString(mention)
    entityIdMap.getOrElseUpdate(string, entityIdMap.size.toLong)
  }

  def addMention(mention: Mention, id: Long) {
    addMention(id, getEntityId(mention))
  }

  def name: String

  override def toString: String =
    ("--- %s ---\n" +
          "num mentions: %d\n" +
          "num entities: %d").format(name, numMentions, numEntities)

  def sizeHistogram: HashMap[Int, Int] = {
    val counts = new HashMap[Int, Int]
    for (entity <- getEntities) {
      val size = entity.size
      counts(size) = counts.getOrElse(size, 0) + 1
    }
    counts
  }

  def writeSizeHistogramToFile(outputFilename: String) {
    val hist = sizeHistogram
    val sizes = hist.keys.toSeq.sorted
    // write the data
    val writer = new PrintWriter("%s-%s.sizes.hist.dat".format(outputFilename, name))
    for (size <- sizes) {
      writer.println("%d\t%d".format(size, hist(size)))
    }
    writer.flush()
    writer.close()
    //    val xySeries = Plotting.xySeries(sizes.map(_.toDouble), sizes.map(hist(_).toDouble))
    //    // plot the file
    //    val xyColl = new XYSeriesCollection
    //    xyColl.addSeries(xySeries)
    //    val chart = Plotting.createChart(xyColl, "Size Histogram", "Entity Size", "", false) // (%s)".format(name)
    //    val plot = chart.getXYPlot
    //    plot.setDomainAxis(new LogarithmicAxis("Entity Sizes"))
    //    plot.setRangeAxis(new LogarithmicAxis("Number of Entities"))
    //    val renderer = new XYLineAndShapeRenderer()
    //    renderer.setSeriesLinesVisible(0, false)
    //    renderer.setSeriesShapesVisible(0, true)
    //    renderer.setSeriesPaint(0, new Color(0, 0, 255, 50))
    //    renderer.setSeriesShape(0, new Ellipse2D.Float(2f, 2f, 2f, 2f))
    //    //renderer.setSeriesFillPaint(0, colors(0 % colors.length))
    //    plot.setRenderer(renderer)
    //    Plotting.writeToPDF("%s-%s.sizes.hist.pdf".format(outputFilename, name), chart, 400, 300)
  }

  def writeToFile(outputFilename: String) {
    val writer = new PrintWriter("%s-%s.map".format(outputFilename, name))
    for ((mid, eid) <- reverseMap)
      writer.println("%d\t%d".format(mid, eid))
    writer.flush()
    writer.close()
  }

  def writeSizesToFile(outputFilename: String) {
    val writer = new PrintWriter("%s-%s.map".format(outputFilename, name))
    for ((entityStr, eid) <- entityIdMap)
      writer.println("%d\t%s".format(getMentions(eid).size, entityStr))
    writer.flush()
    writer.close()
  }

  def copy(minFilter: Int, maxFilter: Int): EntityMap = {
    val map = new EntityMap
    var entityId = 0
    for (entity <- getEntities) {
      if (entity.size >= minFilter && entity.size <= maxFilter) {
        for (mid <- entity) map.addMention(mid, entityId)
        entityId += 1
      }
    }
    map
  }

  def copy(truth: EntityMap): EntityMap = {
    val map = new EntityMap
    for (mid <- truth.getMentionIds) {
      map.addMention(mid, getEntity(mid))
    }
    map
  }
}

class WikiURLMap extends Clustering {
  def getEntityString(m: Mention) = m.url.replaceAll("^shttp", "http")

  def name = "WikiURLMap"
}

class WikiPathMap extends Clustering {
  def getEntityString(m: Mention) = new URL(m.url.replaceAll("^shttp", "http")).getPath

  def name = "WikiPathMap"
}

class AnchorTextMap extends Clustering {
  def getEntityString(mention: Mention) = mention.text

  def name = "AnchorTextMap"
}

class LowerAnchorTextMap extends Clustering {
  def getEntityString(mention: Mention) = mention.text.toLowerCase

  def name = "LowerAnchorMap"
}

class RightmostLowerTokenMap extends Clustering {
  def getEntityString(mention: Mention) = mention.text.split("\\b").last

  def name = "RightmostMap"
}

object Baselines {
  def main(args: Array[String]) {
    val opts = CmdLine.parse(args)
    println(opts)
    val dirName = opts.get("dir")
    val output = opts.getOrElse("output", "/Users/sameer/tmp/output")
    val takeOnly = opts.getOrElse("take", Int.MaxValue.toString).toInt
    val minFilter = opts.getOrElse("min", "1").toInt
    val maxFilter = opts.getOrElse("max", Int.MaxValue.toString).toInt

    if (dirName.isEmpty) {
      println("Directory not included. Use @dir=/path/to/google/gz/files/")
      sys.exit(1)
    }

    TimeUtil.init
    //
    val wikiURLMap = new WikiURLMap
    val wikiPathMap = new WikiPathMap
    val truth = Seq(wikiURLMap) //, wikiPathMap)

    val anchorTextMap = new AnchorTextMap
    val lowerTextMap = new LowerAnchorTextMap
    val rightmostMap = new RightmostLowerTokenMap
    val pred = Seq(anchorTextMap, lowerTextMap, rightmostMap)

    var pageId = 0
    var mentionId: Long = 0
    for (page <- WebpageIterator(dirName.get).take(takeOnly)) {
      if (pageId % 100000 == 0) {
        TimeUtil.snapshot("Done %d".format(pageId))
      }
      for (mention <- page.mentions) {
        truth.foreach(_.addMention(mention, mentionId))
        pred.foreach(_.addMention(mention, mentionId))
        mentionId += 1
      }
      pageId += 1
    }
    TimeUtil.snapshot("Maps built")
    for (t <- truth) {
      val tcopy = t.copy(minFilter, maxFilter)
      TimeUtil.snapshot("Copied truth")
      pred.foreach(p => {
        val pcopy = p.copy(tcopy)
        println(t.toString + "\n" + p.toString + "\n" + CorefEvaluator.evaluate(pcopy, tcopy, true))
      })
    }
    TimeUtil.snapshot("Evaluated")
    truth.foreach(t => {
      t.writeToFile(output)
      TimeUtil.snapshot("Written " + t.name)
    })
    pred.foreach(p => {
      p.writeToFile(output)
      TimeUtil.snapshot("Written " + p.name)
    })
    TimeUtil.snapshot("Maps written to files")
    truth.foreach(t => t.writeSizeHistogramToFile(output))
    TimeUtil.snapshot("Histogram written")
  }
}