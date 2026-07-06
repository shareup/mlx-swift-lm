import Foundation
import MLX

package enum Gemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    package var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            keys.0.dim(-2)
        }
    }
}

package enum Gemma4ToolSchemaNormalizer {
    package static func normalizeTypeArrays(_ tools: [ToolSpec]?) throws -> [ToolSpec]? {
        guard let tools else { return nil }

        return try tools.map { tool in
            try normalize(tool)
        }
    }

    private static func normalize(_ tool: ToolSpec) throws -> ToolSpec {
        guard var function = tool["function"] as? [String: any Sendable] else {
            return tool
        }

        let toolName = function["name"] as? String
        if let parameters = function["parameters"] as? [String: any Sendable] {
            function["parameters"] = try normalizeSchema(
                parameters, path: "function.parameters", toolName: toolName)
        }
        if let response = function["response"] as? [String: any Sendable] {
            function["response"] = try normalizeSchema(
                response, path: "function.response", toolName: toolName)
        }

        var tool = tool
        tool["function"] = function
        return tool
    }

    private static func normalizeSchema(
        _ schema: [String: any Sendable], path: String, toolName: String?
    ) throws -> [String: any Sendable] {
        var schema = try normalizeType(in: schema, path: path, toolName: toolName)

        if let properties = schema["properties"] as? [String: any Sendable] {
            schema["properties"] = try normalizeSchemaMap(
                properties, path: "\(path).properties", toolName: toolName)
        }

        if let items = schema["items"] as? [String: any Sendable] {
            schema["items"] = try normalizeSchema(items, path: "\(path).items", toolName: toolName)
        } else if let itemValue = schema["items"], let items = schemaArray(from: itemValue) {
            schema["items"] = try items.enumerated().map { index, item in
                try normalizeSchema(item, path: "\(path).items[\(index)]", toolName: toolName)
            }
        }

        if let additionalProperties = schema["additionalProperties"] as? [String: any Sendable] {
            schema["additionalProperties"] = try normalizeSchema(
                additionalProperties, path: "\(path).additionalProperties", toolName: toolName)
        }

        for key in ["$defs", "definitions"] {
            if let definitions = schema[key] as? [String: any Sendable] {
                schema[key] = try normalizeSchemaMap(
                    definitions, path: "\(path).\(key)", toolName: toolName)
            }
        }

        for key in ["anyOf", "oneOf", "allOf"] {
            if let choiceValue = schema[key], let choices = schemaArray(from: choiceValue) {
                schema[key] = try choices.enumerated().map { index, choice in
                    try normalizeSchema(
                        choice, path: "\(path).\(key)[\(index)]", toolName: toolName)
                }
            }
        }

        return schema
    }

    private static func normalizeSchemaMap(
        _ schemas: [String: any Sendable], path: String, toolName: String?
    ) throws -> [String: any Sendable] {
        var normalized: [String: any Sendable] = [:]
        normalized.reserveCapacity(schemas.count)
        for (name, value) in schemas {
            if let schema = value as? [String: any Sendable] {
                normalized[name] = try normalizeSchema(
                    schema, path: "\(path).\(name)", toolName: toolName)
            } else {
                normalized[name] = value
            }
        }
        return normalized
    }

    private static func normalizeType(
        in schema: [String: any Sendable], path: String, toolName: String?
    ) throws -> [String: any Sendable] {
        guard let typeValue = schema["type"] else {
            return schema
        }
        guard isArray(typeValue) else {
            return schema
        }
        guard let types = stringArray(from: typeValue) else {
            throw Gemma4ToolSchemaNormalizationError(
                path: "\(path).type",
                toolName: toolName,
                typeDescription: String(describing: typeValue))
        }
        guard !types.isEmpty else {
            throw Gemma4ToolSchemaNormalizationError(
                path: "\(path).type",
                toolName: toolName,
                typeDescription: format(types))
        }

        let nonNullTypes = types.filter { $0.lowercased() != "null" }
        guard Set(nonNullTypes.map { $0.lowercased() }).count <= 1 else {
            throw Gemma4ToolSchemaNormalizationError(
                path: "\(path).type",
                toolName: toolName,
                typeDescription: format(types))
        }

        var schema = schema
        if let type = nonNullTypes.first {
            schema["type"] = type
        } else if let type = types.first {
            schema["type"] = type
        }
        if types.contains(where: { $0.lowercased() == "null" }) {
            schema["nullable"] = true
        }

        return schema
    }

    private static func isArray(_ value: any Sendable) -> Bool {
        value is [String] || value is [any Sendable]
    }

    private static func stringArray(from value: any Sendable) -> [String]? {
        if let strings = value as? [String] {
            return strings
        }
        if let values = value as? [any Sendable] {
            var strings = [String]()
            for value in values {
                guard let string = value as? String else {
                    return nil
                }
                strings.append(string)
            }
            return strings
        }
        return nil
    }

    private static func schemaArray(from value: any Sendable) -> [[String: any Sendable]]? {
        if let schemas = value as? [[String: any Sendable]] {
            return schemas
        }
        if let values = value as? [any Sendable] {
            var schemas = [[String: any Sendable]]()
            for value in values {
                guard let schema = value as? [String: any Sendable] else {
                    return nil
                }
                schemas.append(schema)
            }
            return schemas
        }
        return nil
    }

    private static func format(_ strings: [String]) -> String {
        "[" + strings.map { #""\#($0)""# }.joined(separator: ", ") + "]"
    }
}

private struct Gemma4ToolSchemaNormalizationError: LocalizedError {
    let path: String
    let toolName: String?
    let typeDescription: String

    var errorDescription: String? {
        let toolDescription = toolName.map { " for tool \($0)" } ?? ""
        return
            "Gemma4 tool schema cannot render union type\(toolDescription) at \(path): \(typeDescription)"
    }
}
