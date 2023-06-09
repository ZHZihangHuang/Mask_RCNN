# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/distributed_runtime_payloads.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/protobuf/distributed_runtime_payloads.proto',
  package='tensorflow.distributed_runtime',
  syntax='proto3',
  serialized_options=_b('ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\370\001\001'),
  serialized_pb=_b('\n;tensorflow/core/protobuf/distributed_runtime_payloads.proto\x12\x1etensorflow.distributed_runtime\"\x9d\x01\n\x14GrpcPayloadContainer\x12T\n\x08payloads\x18\x01 \x03(\x0b\x32\x42.tensorflow.distributed_runtime.GrpcPayloadContainer.PayloadsEntry\x1a/\n\rPayloadsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\"\x12\n\x10GrpcPayloadsLost\"\x19\n\x17WorkerPossiblyRestartedBZZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\xf8\x01\x01\x62\x06proto3')
)




_GRPCPAYLOADCONTAINER_PAYLOADSENTRY = _descriptor.Descriptor(
  name='PayloadsEntry',
  full_name='tensorflow.distributed_runtime.GrpcPayloadContainer.PayloadsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.distributed_runtime.GrpcPayloadContainer.PayloadsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.distributed_runtime.GrpcPayloadContainer.PayloadsEntry.value', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=206,
  serialized_end=253,
)

_GRPCPAYLOADCONTAINER = _descriptor.Descriptor(
  name='GrpcPayloadContainer',
  full_name='tensorflow.distributed_runtime.GrpcPayloadContainer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='payloads', full_name='tensorflow.distributed_runtime.GrpcPayloadContainer.payloads', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GRPCPAYLOADCONTAINER_PAYLOADSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=96,
  serialized_end=253,
)


_GRPCPAYLOADSLOST = _descriptor.Descriptor(
  name='GrpcPayloadsLost',
  full_name='tensorflow.distributed_runtime.GrpcPayloadsLost',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=255,
  serialized_end=273,
)


_WORKERPOSSIBLYRESTARTED = _descriptor.Descriptor(
  name='WorkerPossiblyRestarted',
  full_name='tensorflow.distributed_runtime.WorkerPossiblyRestarted',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=275,
  serialized_end=300,
)

_GRPCPAYLOADCONTAINER_PAYLOADSENTRY.containing_type = _GRPCPAYLOADCONTAINER
_GRPCPAYLOADCONTAINER.fields_by_name['payloads'].message_type = _GRPCPAYLOADCONTAINER_PAYLOADSENTRY
DESCRIPTOR.message_types_by_name['GrpcPayloadContainer'] = _GRPCPAYLOADCONTAINER
DESCRIPTOR.message_types_by_name['GrpcPayloadsLost'] = _GRPCPAYLOADSLOST
DESCRIPTOR.message_types_by_name['WorkerPossiblyRestarted'] = _WORKERPOSSIBLYRESTARTED
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GrpcPayloadContainer = _reflection.GeneratedProtocolMessageType('GrpcPayloadContainer', (_message.Message,), {

  'PayloadsEntry' : _reflection.GeneratedProtocolMessageType('PayloadsEntry', (_message.Message,), {
    'DESCRIPTOR' : _GRPCPAYLOADCONTAINER_PAYLOADSENTRY,
    '__module__' : 'tensorflow.core.protobuf.distributed_runtime_payloads_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.distributed_runtime.GrpcPayloadContainer.PayloadsEntry)
    })
  ,
  'DESCRIPTOR' : _GRPCPAYLOADCONTAINER,
  '__module__' : 'tensorflow.core.protobuf.distributed_runtime_payloads_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.distributed_runtime.GrpcPayloadContainer)
  })
_sym_db.RegisterMessage(GrpcPayloadContainer)
_sym_db.RegisterMessage(GrpcPayloadContainer.PayloadsEntry)

GrpcPayloadsLost = _reflection.GeneratedProtocolMessageType('GrpcPayloadsLost', (_message.Message,), {
  'DESCRIPTOR' : _GRPCPAYLOADSLOST,
  '__module__' : 'tensorflow.core.protobuf.distributed_runtime_payloads_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.distributed_runtime.GrpcPayloadsLost)
  })
_sym_db.RegisterMessage(GrpcPayloadsLost)

WorkerPossiblyRestarted = _reflection.GeneratedProtocolMessageType('WorkerPossiblyRestarted', (_message.Message,), {
  'DESCRIPTOR' : _WORKERPOSSIBLYRESTARTED,
  '__module__' : 'tensorflow.core.protobuf.distributed_runtime_payloads_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.distributed_runtime.WorkerPossiblyRestarted)
  })
_sym_db.RegisterMessage(WorkerPossiblyRestarted)


DESCRIPTOR._options = None
_GRPCPAYLOADCONTAINER_PAYLOADSENTRY._options = None
# @@protoc_insertion_point(module_scope)