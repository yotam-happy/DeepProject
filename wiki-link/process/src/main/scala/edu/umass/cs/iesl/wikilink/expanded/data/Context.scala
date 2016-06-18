package edu.umass.cs.iesl.wikilink.expanded.data

import com.twitter.scrooge.{ThriftStruct, ThriftStructCodec}
import org.apache.thrift.protocol._
import java.nio.ByteBuffer
import com.twitter.finagle.SourcedException
import scala.collection.mutable
import scala.collection.{Map, Set}

object Context extends ThriftStructCodec[Context] {
  val Struct = new TStruct("Context")
  val LeftField = new TField("left", TType.STRING, 1)
  val RightField = new TField("right", TType.STRING, 2)
  val MiddleField = new TField("middle", TType.STRING, 3)

  def encode(_item: Context, _oproto: TProtocol) { _item.write(_oproto) }
  def decode(_iprot: TProtocol) = Immutable.decode(_iprot)

  def apply(_iprot: TProtocol): Context = decode(_iprot)

  def apply(
    `left`: String,
    `right`: String,
    `middle`: String
  ): Context = new Immutable(
    `left`,
    `right`,
    `middle`
  )

  def unapply(_item: Context): Option[Product3[String, String, String]] = Some(_item)

  object Immutable extends ThriftStructCodec[Context] {
    def encode(_item: Context, _oproto: TProtocol) { _item.write(_oproto) }
    def decode(_iprot: TProtocol) = {
      var `left`: String = null
      var _got_left = false
      var `right`: String = null
      var _got_right = false
      var `middle`: String = null
      var _got_middle = false
      var _done = false
      _iprot.readStructBegin()
      while (!_done) {
        val _field = _iprot.readFieldBegin()
        if (_field.`type` == TType.STOP) {
          _done = true
        } else {
          _field.id match {
            case 1 => { /* left */
              _field.`type` match {
                case TType.STRING => {
                  `left` = {
                    _iprot.readString()
                  }
                  _got_left = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 2 => { /* right */
              _field.`type` match {
                case TType.STRING => {
                  `right` = {
                    _iprot.readString()
                  }
                  _got_right = true
                }
                case _ => TProtocolUtil.skip(_iprot, _field.`type`)
              }
            }
            case 3 => { /* middle */
              _field.`type` match {
                case TType.STRING => {
                  `middle` = {
                    _iprot.readString()
                  }
                  _got_middle = true
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
        `left`,
        `right`,
        `middle`
      )
    }
  }

  /**
   * The default read-only implementation of Context.  You typically should not need to
   * directly reference this class; instead, use the Context.apply method to construct
   * new instances.
   */
  class Immutable(
    val `left`: String,
    val `right`: String,
    val `middle`: String
  ) extends Context

  /**
   * This Proxy trait allows you to extend the Context trait with additional state or
   * behavior and implement the read-only methods from Context using an underlying
   * instance.
   */
  trait Proxy extends Context {
    protected def _underlyingContext: Context
    def `left`: String = _underlyingContext.`left`
    def `right`: String = _underlyingContext.`right`
    def `middle`: String = _underlyingContext.`middle`
  }
}

trait Context extends ThriftStruct
  with Product3[String, String, String]
  with java.io.Serializable
{
  import Context._

  def `left`: String
  def `right`: String
  def `middle`: String

  def _1 = `left`
  def _2 = `right`
  def _3 = `middle`

  override def write(_oprot: TProtocol) {
    validate()
    _oprot.writeStructBegin(Struct)
    if (true) {
      val `left_item` = `left`
      _oprot.writeFieldBegin(LeftField)
      _oprot.writeString(`left_item`)
      _oprot.writeFieldEnd()
    }
    if (true) {
      val `right_item` = `right`
      _oprot.writeFieldBegin(RightField)
      _oprot.writeString(`right_item`)
      _oprot.writeFieldEnd()
    }
    if (true) {
      val `middle_item` = `middle`
      _oprot.writeFieldBegin(MiddleField)
      _oprot.writeString(`middle_item`)
      _oprot.writeFieldEnd()
    }
    _oprot.writeFieldStop()
    _oprot.writeStructEnd()
  }

  def copy(
    `left`: String = this.`left`,
    `right`: String = this.`right`,
    `middle`: String = this.`middle`
  ): Context = new Immutable(
    `left`,
    `right`,
    `middle`
  )

  /**
   * Checks that all required fields are non-null.
   */
  def validate() {
  }

  def canEqual(other: Any) = other.isInstanceOf[Context]

  override def equals(other: Any): Boolean = runtime.ScalaRunTime._equals(this, other)

  override def hashCode: Int = runtime.ScalaRunTime._hashCode(this)

  override def toString: String = runtime.ScalaRunTime._toString(this)

  override def productArity = 3

  override def productElement(n: Int): Any = n match {
    case 0 => `left`
    case 1 => `right`
    case 2 => `middle`
    case _ => throw new IndexOutOfBoundsException(n.toString)
  }

  override def productPrefix = "Context"
}