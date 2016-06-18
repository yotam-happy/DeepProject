package edu.umass.cs.iesl.wikilink.expanded.data

import com.twitter.scrooge.{ThriftStruct, ThriftStructCodec}
import org.apache.thrift.protocol._
import java.nio.ByteBuffer
import com.twitter.finagle.SourcedException
import scala.collection.mutable
import scala.collection.{Map, Set}

object WikiLinkItem extends ThriftStructCodec[WikiLinkItem] {
  val Struct = new TStruct("WikiLinkItem")
  val DocIdField = new TField("docId", TType.I32, 1)
  val UrlField = new TField("url", TType.STRING, 2)
  val ContentField = new TField("content", TType.STRUCT, 3)
  val RareWordsField = new TField("rareWords", TType.LIST, 4)
  val MentionsField = new TField("mentions", TType.LIST, 5)

  def encode(_item: WikiLinkItem, _oproto: TProtocol) { _item.write(_oproto) }
  def decode(_iprot: TProtocol) = Immutable.decode(_iprot)

  def apply(_iprot: TProtocol): WikiLinkItem = decode(_iprot)

  def apply(
    `docId`: Int,
    `url`: String,
    `content`: PageContentItem,
    `rareWords`: Seq[RareWord] = Seq[RareWord](),
    `mentions`: Seq[Mention] = Seq[Mention]()
  ): WikiLinkItem = new Immutable(
    `docId`,
    `url`,
    `content`,
    `rareWords`,
    `mentions`
  )

  def unapply(_item: WikiLinkItem): Option[Product5[Int, String, PageContentItem, Seq[RareWord], Seq[Mention]]] = Some(_item)

  object Immutable extends ThriftStructCodec[WikiLinkItem] {
    def encode(_item: WikiLinkItem, _oproto: TProtocol) { _item.write(_oproto) }
    def decode(_iprot: TProtocol) = {
      var `docId`: Int = 0
      var _got_docId = false
      var `url`: String = null
      var _got_url = false
      var `content`: PageContentItem = null
      var _got_content = false
      var `rareWords`: Seq[RareWord] = Seq[RareWord]()
      var _got_rareWords = false
      var `mentions`: Seq[Mention] = Seq[Mention]()
      var _got_mentions = false
      var _done = false
      _iprot.readStructBegin()
      while (!_done) {
        val _field = _iprot.readFieldBegin()
        if (_field.`type` == TType.STOP) {
          _done = true
        } else {
          _field.id match {
            case 1 => { /* docId */
              _field.`type` match {
                case TType.I32 => {
                  `docId` = {
                    _iprot.readI32()
                  }
                  _got_docId = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 2 => { /* url */
              _field.`type` match {
                case TType.STRING => {
                  `url` = {
                    _iprot.readString()
                  }
                  _got_url = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 3 => { /* content */
              _field.`type` match {
                case TType.STRUCT => {
                  `content` = {
                    PageContentItem.decode(_iprot)
                  }
                  _got_content = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 4 => { /* rareWords */
              _field.`type` match {
                case TType.LIST => {
                  `rareWords` = {
                    val _list = _iprot.readListBegin()
                    val _rv = new mutable.ArrayBuffer[RareWord](_list.size)
                    var _i = 0
                    while (_i < _list.size) {
                      _rv += {
                        RareWord.decode(_iprot)
                      }
                      _i += 1
                    }
                    _iprot.readListEnd()
                    _rv
                  }
                  _got_rareWords = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 5 => { /* mentions */
              _field.`type` match {
                case TType.LIST => {
                  `mentions` = {
                    val _list = _iprot.readListBegin()
                    val _rv = new mutable.ArrayBuffer[Mention](_list.size)
                    var _i = 0
                    while (_i < _list.size) {
                      _rv += {
                        Mention.decode(_iprot)
                      }
                      _i += 1
                    }
                    _iprot.readListEnd()
                    _rv
                  }
                  _got_mentions = true
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
        `docId`,
        `url`,
        `content`,
        `rareWords`,
        `mentions`
      )
    }
  }

  /**
   * The default read-only implementation of WikiLinkItem.  You typically should not need to
   * directly reference this class; instead, use the WikiLinkItem.apply method to construct
   * new instances.
   */
  class Immutable(
    val `docId`: Int,
    val `url`: String,
    val `content`: PageContentItem,
    val `rareWords`: Seq[RareWord] = Seq[RareWord](),
    val `mentions`: Seq[Mention] = Seq[Mention]()
  ) extends WikiLinkItem

  /**
   * This Proxy trait allows you to extend the WikiLinkItem trait with additional state or
   * behavior and implement the read-only methods from WikiLinkItem using an underlying
   * instance.
   */
  trait Proxy extends WikiLinkItem {
    protected def _underlyingWikiLinkItem: WikiLinkItem
    def `docId`: Int = _underlyingWikiLinkItem.`docId`
    def `url`: String = _underlyingWikiLinkItem.`url`
    def `content`: PageContentItem = _underlyingWikiLinkItem.`content`
    def `rareWords`: Seq[RareWord] = _underlyingWikiLinkItem.`rareWords`
    def `mentions`: Seq[Mention] = _underlyingWikiLinkItem.`mentions`
  }
}

trait WikiLinkItem extends ThriftStruct
  with Product5[Int, String, PageContentItem, Seq[RareWord], Seq[Mention]]
  with java.io.Serializable
{
  import WikiLinkItem._

  def `docId`: Int
  def `url`: String
  def `content`: PageContentItem
  def `rareWords`: Seq[RareWord]
  def `mentions`: Seq[Mention]

  def _1 = `docId`
  def _2 = `url`
  def _3 = `content`
  def _4 = `rareWords`
  def _5 = `mentions`

  override def write(_oprot: TProtocol) {
    validate()
    _oprot.writeStructBegin(Struct)
    if (true) {
      val `docId_item` = `docId`
      _oprot.writeFieldBegin(DocIdField)
      _oprot.writeI32(`docId_item`)
      _oprot.writeFieldEnd()
    }
    if (true) {
      val `url_item` = `url`
      _oprot.writeFieldBegin(UrlField)
      _oprot.writeString(`url_item`)
      _oprot.writeFieldEnd()
    }
    if (true) {
      val `content_item` = `content`
      _oprot.writeFieldBegin(ContentField)
      `content_item`.write(_oprot)
      _oprot.writeFieldEnd()
    }
    if (true) {
      val `rareWords_item` = `rareWords`
      _oprot.writeFieldBegin(RareWordsField)
      _oprot.writeListBegin(new TList(TType.STRUCT, `rareWords_item`.size))
      `rareWords_item`.foreach { `_rareWords_item_element` =>
        `_rareWords_item_element`.write(_oprot)
      }
      _oprot.writeListEnd()
      _oprot.writeFieldEnd()
    }
    if (true) {
      val `mentions_item` = `mentions`
      _oprot.writeFieldBegin(MentionsField)
      _oprot.writeListBegin(new TList(TType.STRUCT, `mentions_item`.size))
      `mentions_item`.foreach { `_mentions_item_element` =>
        `_mentions_item_element`.write(_oprot)
      }
      _oprot.writeListEnd()
      _oprot.writeFieldEnd()
    }
    _oprot.writeFieldStop()
    _oprot.writeStructEnd()
  }

  def copy(
    `docId`: Int = this.`docId`,
    `url`: String = this.`url`,
    `content`: PageContentItem = this.`content`,
    `rareWords`: Seq[RareWord] = this.`rareWords`,
    `mentions`: Seq[Mention] = this.`mentions`
  ): WikiLinkItem = new Immutable(
    `docId`,
    `url`,
    `content`,
    `rareWords`,
    `mentions`
  )

  /**
   * Checks that all required fields are non-null.
   */
  def validate() {
  }

  def canEqual(other: Any) = other.isInstanceOf[WikiLinkItem]

  override def equals(other: Any): Boolean = runtime.ScalaRunTime._equals(this, other)

  override def hashCode: Int = runtime.ScalaRunTime._hashCode(this)

  override def toString: String = runtime.ScalaRunTime._toString(this)

  override def productArity = 5

  override def productElement(n: Int): Any = n match {
    case 0 => `docId`
    case 1 => `url`
    case 2 => `content`
    case 3 => `rareWords`
    case 4 => `mentions`
    case _ => throw new IndexOutOfBoundsException(n.toString)
  }

  override def productPrefix = "WikiLinkItem"
}