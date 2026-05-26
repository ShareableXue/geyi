Return exactly one JSON object:
{
  "intent_confirmation": "string",
  "selected_backend": "tilelang|ascendc",
  "selected_template": "string or null",
  "parameter_bindings": {},
  "required_assumptions": ["string"],
  "risks": ["string"],
  "repair_suggestions": ["string"],
  "cannot_translate": false,
  "annotation_request": {"question": "string", "unknown_ids": ["string"]}
}
