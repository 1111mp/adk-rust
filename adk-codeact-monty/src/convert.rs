//! Marshalling between the framework's JSON values and Monty's [`MontyObject`].
//!
//! The CodeAct seam speaks `serde_json::Value`: a [`PendingCall`] surfaces tool
//! arguments as JSON, the driver feeds a tool's JSON result back in, and a
//! finished script reports its value as JSON. Monty speaks [`MontyObject`]. These
//! two functions bridge the gap.
//!
//! [`PendingCall`]: adk_agent::codeact::PendingCall

use monty_types::{DictPairs, MontyObject};
use serde_json::{Map, Number, Value};

/// Convert a host JSON value into a Monty value, to be injected into a script
/// (a tool result, a resolved name, ...).
///
/// The mapping is the obvious one. A JSON integer that fits in `i64` becomes an
/// `int`; anything larger (or fractional) becomes a `float`. Objects become
/// `dict`s keyed by their string keys, preserving insertion order.
pub(crate) fn json_to_monty(value: Value) -> MontyObject {
    match value {
        Value::Null => MontyObject::None,
        Value::Bool(b) => MontyObject::Bool(b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                MontyObject::Int(i)
            } else {
                // u64 values above i64::MAX and all non-integral numbers fall
                // back to float — good enough for tool I/O, which is rarely
                // built around > 2^63 integers.
                MontyObject::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(s) => MontyObject::String(s),
        Value::Array(items) => MontyObject::List(items.into_iter().map(json_to_monty).collect()),
        Value::Object(map) => {
            let pairs: Vec<(MontyObject, MontyObject)> =
                map.into_iter().map(|(k, v)| (MontyObject::String(k), json_to_monty(v))).collect();
            MontyObject::Dict(DictPairs::from(pairs))
        }
    }
}

/// Convert a Monty value produced by a script into host JSON.
///
/// This is the "natural" projection: JSON-native Python values map to their
/// bare JSON form (`42`, `"hi"`, `[...]`, `{"a": 1}`). It is deliberately *not*
/// `serde_json::to_value(obj)` — [`MontyObject`]'s derived `Serialize` is an
/// externally tagged snapshot format (`{"Int": 42}`) meant for binary
/// transport, not human-facing JSON. The rare non-JSON-native value (a
/// `tuple`, `bytes`, a `date`, ...) degrades to its Python `repr` string; the
/// CodeAct contract only requires the top-level completion value to be a
/// tagged object, which a script's final `dict` expression always is.
pub(crate) fn monty_to_json(obj: &MontyObject) -> Value {
    match obj {
        MontyObject::None => Value::Null,
        MontyObject::Bool(b) => Value::Bool(*b),
        MontyObject::Int(i) => Value::Number(Number::from(*i)),
        // NaN/Infinity have no JSON form; degrade to null rather than fail.
        MontyObject::Float(f) => Number::from_f64(*f).map_or(Value::Null, Value::Number),
        MontyObject::String(s) => Value::String(s.clone()),
        MontyObject::Path(p) => Value::String(p.clone()),
        // Every Python sequence projects to a JSON array.
        MontyObject::List(items)
        | MontyObject::Tuple(items)
        | MontyObject::Set(items)
        | MontyObject::FrozenSet(items) => Value::Array(items.iter().map(monty_to_json).collect()),
        MontyObject::Dict(pairs) => dict_to_json(pairs),
        // Everything else (bytes, datetimes, exceptions, file handles, ...) has
        // no JSON-native shape, so it degrades to its Python `repr`. These never
        // appear as a top-level CodeAct completion value.
        other => Value::String(other.py_repr()),
    }
}

/// Project a Monty `dict` to a JSON object, stringifying any non-string key
/// via its Python `repr` (JSON object keys must be strings; script
/// argument/result dicts are string-keyed in practice).
fn dict_to_json(pairs: &DictPairs) -> Value {
    let mut map = Map::new();
    for (key, value) in pairs {
        let key = match key {
            MontyObject::String(s) => s.clone(),
            other => other.py_repr(),
        };
        map.insert(key, monty_to_json(value));
    }
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trips_json_native_values() {
        let cases = [
            json!(null),
            json!(true),
            json!(42),
            json!(3.5),
            json!("hello"),
            json!([1, 2, 3]),
            json!({"type": "final_result", "value": {"n": 7}}),
        ];
        for case in cases {
            let monty = json_to_monty(case.clone());
            assert_eq!(monty_to_json(&monty), case);
        }
    }

    #[test]
    fn large_u64_degrades_to_float() {
        let big = json!(u64::MAX);
        assert!(matches!(json_to_monty(big), MontyObject::Float(_)));
    }
}
