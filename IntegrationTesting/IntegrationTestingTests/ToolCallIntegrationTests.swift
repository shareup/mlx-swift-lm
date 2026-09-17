// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

@Suite(.serialized)
struct ToolCallIntegrationTests {

    // MARK: - LFM2

    @Test func lfm2FormatAutoDetection() async throws {
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.lfm2))
        try await ToolCallTests.lfm2FormatAutoDetection(container: container)
    }

    @Test func lfm2EndToEnd() async throws {
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.lfm2))
        try await ToolCallTests.lfm2EndToEndGeneration(container: container)
    }

    @Test func lfm2SupportsJSONArrayNullableToolArguments() async throws {
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.lfm2))
        try await ToolCallTests.lfm2SupportsJSONArrayNullableToolArguments(container: container)
    }

    // MARK: - GLM4

    @Test func glm4FormatAutoDetection() async throws {
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.glm4))
        try await ToolCallTests.glm4FormatAutoDetection(container: container)
    }

    @Test func glm4EndToEnd() async throws {
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.glm4))
        try await ToolCallTests.glm4EndToEndGeneration(container: container)
    }

    @Test func glm4SupportsJSONArrayNullableToolArguments() async throws {
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.glm4))
        try await ToolCallTests.glm4SupportsJSONArrayNullableToolArguments(container: container)
    }

    // MARK: - Mistral3

    @Test func mistral3FormatAutoDetection() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.mistral3))
        try await ToolCallTests.mistral3FormatAutoDetection(container: container)
    }

    @Test func mistral3EndToEnd() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.mistral3))
        try await ToolCallTests.mistral3EndToEndGeneration(container: container)
    }

    @Test func mistral3MultiTool() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.mistral3))
        try await ToolCallTests.mistral3MultiToolGeneration(container: container)
    }

    @Test func mistral3SupportsJSONArrayNullableToolArguments() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.mistral3))
        try await ToolCallTests.mistral3SupportsJSONArrayNullableToolArguments(container: container)
    }

    // MARK: - Nemotron

    @Test func nemotronFormatAutoDetection() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.nemotron))
        try await ToolCallTests.nemotronFormatAutoDetection(container: container)
    }

    @Test func nemotronEndToEnd() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.nemotron))
        try await ToolCallTests.nemotronEndToEndGeneration(container: container)
    }

    @Test func nemotronMultiTool() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.nemotron))
        try await ToolCallTests.nemotronMultiToolGeneration(container: container)
    }

    @Test func nemotronSupportsJSONArrayNullableToolArguments() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.nemotron))
        try await ToolCallTests.nemotronSupportsJSONArrayNullableToolArguments(container: container)
    }

    // MARK: - Qwen3.5

    @Test func qwen35FormatAutoDetection() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.qwen35))
        try await ToolCallTests.qwen35FormatAutoDetection(container: container)
    }

    @Test func qwen35EndToEnd() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.qwen35))
        try await ToolCallTests.qwen35EndToEndGeneration(container: container)
    }

    @Test func qwen35MultiTool() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.qwen35))
        try await ToolCallTests.qwen35MultiToolGeneration(container: container)
    }

    @Test func qwen35SupportsJSONArrayNullableToolArguments() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.qwen35))
        try await ToolCallTests.qwen35SupportsJSONArrayNullableToolArguments(container: container)
    }

    // MARK: - Gemma 4 Unified

    @Test func gemma4UnifiedFormatAutoDetection() async throws {
        let container = try await models.vlmContainer(
            for: .init(id: IntegrationTestModelIDs.gemma4Unified))
        try await ToolCallTests.gemma4FormatAutoDetection(container: container)
    }

    @Test func gemma4UnifiedEndToEnd() async throws {
        let container = try await models.vlmContainer(
            for: .init(id: IntegrationTestModelIDs.gemma4Unified))
        try await ToolCallTests.gemma4EndToEndGeneration(container: container)
    }

    @Test func gemma4UnifiedMultiTool() async throws {
        let container = try await models.vlmContainer(
            for: .init(id: IntegrationTestModelIDs.gemma4Unified))
        try await ToolCallTests.gemma4MultiToolGeneration(container: container)
    }

    @Test func gemma4UnifiedSupportsJSONArrayNullableToolArguments() async throws {
        let container = try await models.vlmContainer(
            for: .init(id: IntegrationTestModelIDs.gemma4Unified))
        try await ToolCallTests.gemma4SupportsJSONArrayNullableToolArguments(container: container)
    }
}
