package edu.umass.cs.iesl.wikilink.expanded.data

import com.twitter.scrooge.{ThriftStruct, ThriftStructCodec}
import org.apache.thrift.protocol._
import java.nio.ByteBuffer
import com.twitter.finagle.SourcedException
import scala.collection.mutable
import scala.collection.{Map, Set}

object PageContentItem extends ThriftStructCodec[PageContentItem] {
  val Struct = new TStruct("PageContentItem")
  val RawField = new TField("raw", TType.STRING, 1)
  val FullTextField = new TField("fullText", TType.STRING, 2)
  val ArticleTextField = new TField("articleText", TType.STRING, 3)
  val DomField = new TField("dom", TType.STRING, 4)

  def encode(_item: PageContentItem, _oproto: TProtocol) { _item.write(_oproto) }
  def decode(_iprot: TProtocol) = Immutable.decode(_iprot)

  def apply(_iprot: TProtocol): PageContentItem = decode(_iprot)

  def apply(
    `raw`: Option[ByteBuffer] = None,
    `fullText`: Option[String] = None,
    `articleText`: Option[String] = None,
    `dom`: Option[String] = None
  ): PageContentItem = new Immutable(
    `raw`,
    `fullText`,
    `articleText`,
    `dom`
  )

  def unapply(_item: PageContentItem): Option[Product4[Option[ByteBuffer], Option[String], Option[String], Option[String]]] = Some(_item)

  object Immutable extends ThriftStructCodec[PageContentItem] {
    def encode(_item: PageContentItem, _oproto: TProtocol) { _item.write(_oproto) }
    def decode(_iprot: TProtocol) = {
      var `raw`: ByteBuffer = null
      var _got_raw = false
      var `fullText`: String = null
      var _got_fullText = false
      var `articleText`: String = null
      var _got_articleText = false
      var `dom`: String = null
      var _got_dom = false
      var _done = false
      _iprot.readStructBegin()
      while (!_done) {
        val _field = _iprot.readFieldBegin()
        if (_field.`type` == TType.STOP) {
          _done = true
        } else {
          _field.id match {
            case 1 => { /* raw */
              _field.`type` match {
                case TType.STRING => {
                  `raw` = {
                    _iprot.readBinary()
                  }
                  _got_raw = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 2 => { /* fullText */
              _field.`type` match {
                case TType.STRING => {
                  `fullText` = {
                    _iprot.readString()
                  }
                  _got_fullText = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 3 => { /* articleText */
              _field.`type` match {
                case TType.STRING => {
                  `articleText` = {
                    _iprot.readString()
                  }
                  _got_articleText = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 4 => { /* dom */
              _field.`type` match {
                case TType.STRING => {
                  `dom` = {
                    _iprot.readString()
                  }
                  _got_dom = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case _ => TProtocolUtil.skip(_iprot, _field.`type`)
          }
          _iprot.readFieldEnd()
        }
      }
      _iprot.readStructEnd()
      new Immutable(
        if (_got_raw) Some(`raw`) else None,
        if (_got_fullText) Some(`fullText`) else None,
        if (_got_articleText) Some(`articleText`) else None,
        if (_got_dom) Some(`dom`) else None
      )
    }
  }

  /**
   * The default read-only implementation of PageContentItem.  You typically should not need to
   * directly reference this class; instead, use the PageContentItem.apply method to construct
   * new instances.
   */
  class Immutable(
    val `raw`: Option[ByteBuffer] = None,
    val `fullText`: Option[String] = None,
    val `articleText`: Option[String] = None,
    val `dom`: Option[String] = None
  ) extends PageContentItem

  /**
   * This Proxy trait allows you to extend the PageContentItem trait with additional state or
   * behavior and implement the read-only methods from PageContentItem using an underlying
   * instance.
   */
  trait Proxy extends PageContentItem {
    protected def _underlyingPageContentItem: PageContentItem
    def `raw`: Option[ByteBuffer] = _underlyingPageContentItem.`raw`
    def `fullText`: Option[String] = _underlyingPageContentItem.`fullText`
    def `articleText`: Option[String] = _underlyingPageContentItem.`articleText`
    def `dom`: Option[String] = _underlyingPageContentItem.`dom`
  }
}

trait PageContentItem extends ThriftStruct
  with Product4[Option[ByteBuffer], Option[String], Option[String], Option[String]]
  with java.io.Serializable
{
  import PageContentItem._

  def `raw`: Option[ByteBuffer]
  def `fullText`: Option[String]
  def `articleText`: Option[String]
  def `dom`: Option[String]

  def _1 = `raw`
  def _2 = `fullText`
  def _3 = `articleText`
  def _4 = `dom`

  override def write(_oprot: TProtocol) {
    validate()
    _oprot.writeStructBegin(Struct)
    if (`raw`.isDefined) {
      val `raw_item` = `raw`.get
      _oprot.writeFieldBegin(RawField)
      _oprot.writeBinary(`raw_item`)
      _oprot.writeFieldEnd()
    }
    if (`fullText`.isDefined) {
      val `fullText_item` = `fullText`.get
      _oprot.writeFieldBegin(FullTextField)
      _oprot.writeString(`fullText_item`)
      _oprot.writeFieldEnd()
    }
    if (`articleText`.isDefined) {
      val `articleText_item` = `articleText`.get
      _oprot.writeFieldBegin(ArticleTextField)
      _oprot.writeString(`articleText_item`)
      _oprot.writeFieldEnd()
    }
    if (`dom`.isDefined) {
      val `dom_item` = `dom`.get
      _oprot.writeFieldBegin(DomField)
      _oprot.writeString(`dom_item`)
      _oprot.writeFieldEnd()
    }
    _oprot.writeFieldStop()
    _oprot.writeStructEnd()
  }

  def copy(
    `raw`: Option[ByteBuffer] = this.`raw`,
    `fullText`: Option[String] = this.`fullText`,
    `articleText`: Option[String] = this.`articleText`,
    `dom`: Option[String] = this.`dom`
  ): PageContentItem = new Immutable(
    `raw`,
    `fullText`,
    `articleText`,
    `dom`
  )

  /**
   * Checks that all required fields are non-null.
   */
  def validate() {
  }

  def canEqual(other: Any) = other.isInstanceOf[PageContentItem]

  override def equals(other: Any): Boolean = runtime.ScalaRunTime._equals(this, other)

  override def hashCode: Int = runtime.ScalaRunTime._hashCode(this)

  override def toString: String = runtime.ScalaRunTime._toString(this)

  override def productArity = 4

  override def productElement(n: Int): Any = n match {
    case 0 => `raw`
    case 1 => `fullText`
    case 2 => `articleText`
    case 3 => `dom`
    case _ => throw new IndexOutOfBoundsException(n.toString)
  }

  override def productPrefix = "PageContentItem"
}