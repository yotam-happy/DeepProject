package edu.umass.cs.iesl.wikilink.expanded.process

import org.apache.thrift.protocol.TBinaryProtocol
import org.apache.thrift.transport.TIOStreamTransport

import java.io.File
import java.io.BufferedOutputStream
import java.io.FileOutputStream
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.util.zip.{GZIPOutputStream, GZIPInputStream}

object ThriftSerializerFactory {
  
  def getWriter(f: File) = {
    val stream = new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(f)), 2048)
    val protocol= new TBinaryProtocol(new TIOStreamTransport(stream))
    (stream, protocol)
  }
  
  def getReader(f: File) = {
    val stream = new BufferedInputStream(new GZIPInputStream(new FileInputStream(f)), 2048)
    val protocol = new TBinaryProtocol(new TIOStreamTransport(stream))
    (stream, protocol)
  }
  
}
