package edu.umass.cs.iesl.wikilink.expanded.data

import com.twitter.scrooge.{ThriftStruct, ThriftStructCodec}
import org.apache.thrift.protocol._
import java.nio.ByteBuffer
import com.twitter.finagle.SourcedException
import scala.collection.mutable
import scala.collection.{Map, Set}

object RareWord extends ThriftStructCodec[RareWord] {
  val Struct = new TStruct("RareWord")
  val WordField = new TField("word", TType.STRING, 1)
  val OffsetField = new TField("offset", TType.I32, 2)

  def encode(_item: RareWord, _oproto: TProtocol) { _item.write(_oproto) }
  def decode(_iprot: TProtocol) = Immutable.decode(_iprot)

  def apply(_iprot: TProtocol): RareWord = decode(_iprot)

  def apply(
    `word`: String,
    `offset`: Int
  ): RareWord = new Immutable(
    `word`,
    `offset`
  )

  def unapply(_item: RareWord): Option[Product2[String, Int]] = Some(_item)

  object Immutable extends ThriftStructCodec[RareWord] {
    def encode(_item: RareWord, _oproto: TProtocol) { _item.write(_oproto) }
    def decode(_iprot: TProtocol) = {
      var `word`: String = null
      var _got_word = false
      var `offset`: Int = 0
      var _got_offset = false
      var _done = false
      _iprot.readStructBegin()
      while (!_done) {
        val _field = _iprot.readFieldBegin()
        if (_field.`type` == TType.STOP) {
          _done = true
        } else {
          _field.id match {
            case 1 => { /* word */
              _field.`type` match {
                case TType.STRING => {
                  `word` = {
                    _iprot.readString()
                  }
                  _got_word = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 2 => { /* offset */
              _field.`type` match {
                case TType.I32 => {
                  `offset` = {
                    _iprot.readI32()
                  }
                  _got_offset = true
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
        `word`,
        `offset`
      )
    }
  }

  /**
   * The default read-only implementation of RareWord.  You typically should not need to
   * directly reference this class; instead, use the RareWord.apply method to construct
   * new instances.
   */
  class Immutable(
    val `word`: String,
    val `offset`: Int
  ) extends RareWord

  /**
   * This Proxy trait allows you to extend the RareWord trait with additional state or
   * behavior and implement the read-only methods from RareWord using an underlying
   * instance.
   */
  trait Proxy extends RareWord {
    protected def _underlyingRareWord: RareWord
    def `word`: String = _underlyingRareWord.`word`
    def `offset`: Int = _underlyingRareWord.`offset`
  }
}

trait RareWord extends ThriftStruct
  with Product2[String, Int]
  with java.io.Serializable
{
  import RareWord._

  def `word`: String
  def `offset`: Int

  def _1 = `word`
  def _2 = `offset`

  override def write(_oprot: TProtocol) {
    validate()
    _oprot.writeStructBegin(Struct)
    if (true) {
      val `word_item` = `word`
      _oprot.writeFieldBegin(WordField)
      _oprot.writeString(`word_item`)
      _oprot.writeFieldEnd()
    }
    if (true) {
      val `offset_item` = `offset`
      _oprot.writeFieldBegin(OffsetField)
      _oprot.writeI32(`offset_item`)
      _oprot.writeFieldEnd()
    }
    _oprot.writeFieldStop()
    _oprot.writeStructEnd()
  }

  def copy(
    `word`: String = this.`word`,
    `offset`: Int = this.`offset`
  ): RareWord = new Immutable(
    `word`,
    `offset`
  )

  /**
   * Checks that all required fields are non-null.
   */
  def validate() {
  }

  def canEqual(other: Any) = other.isInstanceOf[RareWord]

  override def equals(other: Any): Boolean = runtime.ScalaRunTime._equals(this, other)

  override def hashCode: Int = runtime.ScalaRunTime._hashCode(this)

  override def toString: String = runtime.ScalaRunTime._toString(this)

  override def productArity = 2

  override def productElement(n: Int): Any = n match {
    case 0 => `word`
    case 1 => `offset`
    case _ => throw new IndexOutOfBoundsException(n.toString)
  }

  override def productPrefix = "RareWord"
}