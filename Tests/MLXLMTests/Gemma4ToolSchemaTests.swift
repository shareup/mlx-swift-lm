// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXVLM

struct Gemma4ToolSchemaTests {
    @Test("Gemma4 processor passes recursively normalized prompt tools")
    func gemma4ProcessorPassesRecursivelyNormalizedPromptTools() async throws {
        let tool = makeTool(
            properties: [
                "notes": [
                    "type": ["string", "null"],
                    "nullable": true,
                ] as [String: any Sendable],
                "metadata": [
                    "type": ["null", "object"],
                    "nullable": false,
                    "properties": [
                        "source": [
                            "type": ["string", "null"]
                        ] as [String: any Sendable],
                        "tags": [
                            "type": "array",
                            "items": [
                                "type": ["null", "string"]
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "additionalProperties": [
                        "type": ["string", "null"]
                    ] as [String: any Sendable],
                    "$defs": [
                        "label": [
                            "type": ["string", "null"]
                        ] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "definitions": [
                        "count": [
                            "type": ["integer", "null"]
                        ] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "anyOf": [
                        [
                            "type": ["string", "null"]
                        ] as [String: any Sendable]
                    ] as [any Sendable],
                    "oneOf": [
                        [
                            "type": ["number", "null"]
                        ] as [String: any Sendable]
                    ],
                    "allOf": [
                        [
                            "type": ["object", "null"],
                            "properties": [:] as [String: any Sendable],
                        ] as [String: any Sendable]
                    ],
                ] as [String: any Sendable],
            ],
            response: [
                "type": ["object", "null"],
                "properties": [
                    "summary": [
                        "type": ["string", "null"]
                    ] as [String: any Sendable]
                ] as [String: any Sendable],
            ] as [String: any Sendable]
        )

        let capturedTools = try await captureToolsFromGemma4Processor(tool)
        let notes = try property("notes", in: capturedTools[0])
        let metadata = try property("metadata", in: capturedTools[0])
        let metadataProperties = try #require(
            metadata["properties"] as? [String: any Sendable])
        let source = try #require(metadataProperties["source"] as? [String: any Sendable])
        let tags = try #require(metadataProperties["tags"] as? [String: any Sendable])
        let tagItems = try #require(tags["items"] as? [String: any Sendable])
        let additionalProperties = try #require(
            metadata["additionalProperties"] as? [String: any Sendable])
        let defs = try #require(metadata["$defs"] as? [String: any Sendable])
        let label = try #require(defs["label"] as? [String: any Sendable])
        let definitions = try #require(metadata["definitions"] as? [String: any Sendable])
        let count = try #require(definitions["count"] as? [String: any Sendable])
        let anyOf = try #require(metadata["anyOf"] as? [[String: any Sendable]])
        let oneOf = try #require(metadata["oneOf"] as? [[String: any Sendable]])
        let allOf = try #require(metadata["allOf"] as? [[String: any Sendable]])
        let response = try response(in: capturedTools[0])
        let responseProperties = try #require(response["properties"] as? [String: any Sendable])
        let summary = try #require(responseProperties["summary"] as? [String: any Sendable])

        #expect(notes["type"] as? String == "string")
        #expect(notes["nullable"] as? Bool == true)
        #expect(metadata["type"] as? String == "object")
        #expect(metadata["nullable"] as? Bool == true)
        #expect(source["type"] as? String == "string")
        #expect(source["nullable"] as? Bool == true)
        #expect(tags["type"] as? String == "array")
        #expect(tagItems["type"] as? String == "string")
        #expect(tagItems["nullable"] as? Bool == true)
        #expect(additionalProperties["type"] as? String == "string")
        #expect(additionalProperties["nullable"] as? Bool == true)
        #expect(label["type"] as? String == "string")
        #expect(label["nullable"] as? Bool == true)
        #expect(count["type"] as? String == "integer")
        #expect(count["nullable"] as? Bool == true)
        #expect(anyOf[0]["type"] as? String == "string")
        #expect(anyOf[0]["nullable"] as? Bool == true)
        #expect(oneOf[0]["type"] as? String == "number")
        #expect(oneOf[0]["nullable"] as? Bool == true)
        #expect(allOf[0]["type"] as? String == "object")
        #expect(allOf[0]["nullable"] as? Bool == true)
        #expect(response["type"] as? String == "object")
        #expect(response["nullable"] as? Bool == true)
        #expect(summary["type"] as? String == "string")
        #expect(summary["nullable"] as? Bool == true)
    }

    @Test("Gemma4 processor preserves enum, required, and caller schema")
    func gemma4ProcessorPreservesEnumRequiredAndOriginalSchema() async throws {
        let tool = makeTool(
            properties: [
                "status": [
                    "type": "string",
                    "enum": ["ok", NSNull()] as [any Sendable],
                ] as [String: any Sendable],
                "notes": [
                    "type": ["string", "null"]
                ] as [String: any Sendable],
            ],
            required: ["status"]
        )

        let capturedTools = try await captureToolsFromGemma4Processor(tool)
        let parameters = try parameters(in: capturedTools[0])
        let required = try #require(parameters["required"] as? [String])
        let status = try property("status", in: capturedTools[0])
        let enumValues = try #require(status["enum"] as? [any Sendable])
        let originalNotes = try property("notes", in: tool)
        let originalType = try #require(originalNotes["type"] as? [String])

        #expect(required == ["status"])
        #expect(status["type"] as? String == "string")
        #expect(enumValues.count == 2)
        #expect(enumValues[0] as? String == "ok")
        #expect(enumValues[1] is NSNull)
        #expect(originalType == ["string", "null"])
    }

    @Test("Gemma4 processor throws before the tokenizer for unsupported type unions")
    func gemma4ProcessorThrowsBeforeTokenizerForUnsupportedTypeUnions() async throws {
        let tokenizer = Gemma4RecordingTokenizer()
        let processor = Gemma4Processor(try gemma4ProcessorConfiguration(), tokenizer: tokenizer)
        let tool = makeTool(
            name: "lookup",
            properties: [
                "value": [
                    "type": ["string", "number", "null"]
                ] as [String: any Sendable]
            ]
        )

        do {
            _ = try await processor.prepare(input: UserInput(prompt: "hello", tools: [tool]))
            Issue.record("Expected unsupported union to throw before template rendering")
        } catch Gemma4RecordingTokenizerError.stopAfterCapture {
            Issue.record("Expected unsupported union to throw before tokenizer capture")
        } catch {
            #expect(tokenizer.capturedTools == nil)
            #expect(
                error.localizedDescription
                    == #"Gemma4 tool schema cannot render union type for tool lookup at function.parameters.properties.value.type: ["string", "number", "null"]"#
            )
        }
    }

    @Test("Gemma4 unified processor passes normalized prompt tools")
    func gemma4UnifiedProcessorPassesNormalizedPromptTools() async throws {
        let tool = makeTool(
            properties: [
                "notes": [
                    "type": ["string", "null"]
                ] as [String: any Sendable]
            ])

        let capturedTools = try await captureToolsFromGemma4UnifiedProcessor(tool)
        let notes = try property("notes", in: capturedTools[0])
        #expect(notes["type"] as? String == "string")
        #expect(notes["nullable"] as? Bool == true)
    }

    @Test("Gemma4 tool schema generator normalizes nullable prompt tools")
    func gemma4ToolSchemaGeneratorNormalizesNullablePromptTools() throws {
        let tool = makeTool(
            properties: [
                "notes": [
                    "type": ["string", "null"]
                ] as [String: any Sendable]
            ])

        let generatedTools = try #require(
            try Gemma4ToolSchemaGenerator().generate(
                from: UserInput(prompt: "hello", tools: [tool])))
        let notes = try property("notes", in: generatedTools[0])
        #expect(notes["type"] as? String == "string")
        #expect(notes["nullable"] as? Bool == true)
    }

    @Test("Gemma4 tool schema generator throws for unsupported type unions")
    func gemma4ToolSchemaGeneratorThrowsForUnsupportedTypeUnions() throws {
        let tool = makeTool(
            name: "lookup",
            properties: [
                "value": [
                    "type": ["string", "number", "null"]
                ] as [String: any Sendable]
            ]
        )

        do {
            _ = try Gemma4ToolSchemaGenerator().generate(
                from: UserInput(prompt: "hello", tools: [tool]))
            Issue.record("Expected unsupported union to throw")
        } catch {
            #expect(
                error.localizedDescription
                    == #"Gemma4 tool schema cannot render union type for tool lookup at function.parameters.properties.value.type: ["string", "number", "null"]"#
            )
        }
    }
}

private func captureToolsFromGemma4Processor(_ tool: ToolSpec) async throws
    -> [[String: any Sendable]]
{
    let tokenizer = Gemma4RecordingTokenizer()
    let processor = Gemma4Processor(try gemma4ProcessorConfiguration(), tokenizer: tokenizer)

    do {
        _ = try await processor.prepare(input: UserInput(prompt: "hello", tools: [tool]))
        Issue.record("Expected recording tokenizer to stop after capture")
    } catch Gemma4RecordingTokenizerError.stopAfterCapture {
    }

    return try #require(tokenizer.capturedTools)
}

private func captureToolsFromGemma4UnifiedProcessor(_ tool: ToolSpec) async throws
    -> [[String: any Sendable]]
{
    let tokenizer = Gemma4RecordingTokenizer()
    let processor = Gemma4UnifiedProcessor(
        try gemma4UnifiedProcessorConfiguration(), tokenizer: tokenizer)

    do {
        _ = try await processor.prepare(input: UserInput(prompt: "hello", tools: [tool]))
        Issue.record("Expected recording tokenizer to stop after capture")
    } catch Gemma4RecordingTokenizerError.stopAfterCapture {
    }

    return try #require(tokenizer.capturedTools)
}

private func makeTool(
    name: String = "get_weather",
    properties: [String: any Sendable],
    required: [String] = [],
    response: [String: any Sendable]? = nil
) -> ToolSpec {
    var function: [String: any Sendable] = [
        "name": name,
        "description": "Get weather details",
        "parameters": [
            "type": "object",
            "properties": properties,
            "required": required,
        ] as [String: any Sendable],
    ]
    if let response {
        function["response"] = response
    }
    return [
        "type": "function",
        "function": function,
    ]
}

private func parameters(in tool: ToolSpec) throws -> [String: any Sendable] {
    let function = try #require(tool["function"] as? [String: any Sendable])
    return try #require(function["parameters"] as? [String: any Sendable])
}

private func response(in tool: ToolSpec) throws -> [String: any Sendable] {
    let function = try #require(tool["function"] as? [String: any Sendable])
    return try #require(function["response"] as? [String: any Sendable])
}

private func property(_ name: String, in tool: ToolSpec) throws -> [String: any Sendable] {
    let parameters = try parameters(in: tool)
    let properties = try #require(parameters["properties"] as? [String: any Sendable])
    return try #require(properties[name] as? [String: any Sendable])
}

private func gemma4ProcessorConfiguration() throws -> Gemma4ProcessorConfiguration {
    let data = Data(
        """
        {
          "processor_class": "Gemma4Processor",
          "image_token_id": 31,
          "boi_token_id": 28,
          "eoi_token_id": 29
        }
        """.utf8)
    return try JSONDecoder.json5().decode(Gemma4ProcessorConfiguration.self, from: data)
}

private func gemma4UnifiedProcessorConfiguration() throws -> Gemma4UnifiedProcessorConfiguration {
    let data = Data(
        """
        {
          "processor_class": "Gemma4UnifiedProcessor",
          "image_token_id": 31,
          "audio_token_id": 30,
          "video_token_id": 29,
          "boi_token_id": 28,
          "eoi_token_id": 29
        }
        """.utf8)
    return try JSONDecoder.json5().decode(Gemma4UnifiedProcessorConfiguration.self, from: data)
}

private final class Gemma4RecordingTokenizer: Tokenizer, @unchecked Sendable {
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    private(set) var capturedTools: [[String: any Sendable]]?

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [0]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        capturedTools = tools
        throw Gemma4RecordingTokenizerError.stopAfterCapture
    }
}

private enum Gemma4RecordingTokenizerError: Error {
    case stopAfterCapture
}
