from marshmallow import Schema, fields, post_load


class OutputFieldSchema(Schema):
    carbon = fields.List(fields.Float, required=True)
    nitrogen = fields.List(fields.Float, required=True)
    DMon = fields.List(fields.Float, required=True)
    DMoff = fields.List(fields.Float, required=True)


class OutputSchema(Schema):
    above = fields.Nested(OutputFieldSchema)
    below = fields.Nested(OutputFieldSchema)
