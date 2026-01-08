import { streamText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';

export const runtime = 'nodejs';
export const maxDuration = 60;

// Provider factory functions
function getOllamaProvider(model: string, endpoint: string = 'http://localhost:11434') {
  return createOpenAI({
    baseURL: `${endpoint}/v1`,
    apiKey: 'ollama', // Ollama doesn't need a real key but the SDK requires one
  })(model);
}

function getOpenAIProvider(model: string, apiKey: string) {
  return createOpenAI({
    apiKey,
  })(model);
}

function getAnthropicCompatibleProvider(model: string, apiKey: string) {
  // For Anthropic, we'd normally use @ai-sdk/anthropic
  // But since we want to keep deps minimal, we can use OpenAI-compatible proxy
  // or handle it separately. For now, return OpenAI as fallback
  return createOpenAI({
    apiKey,
  })(model);
}

function getGoogleCompatibleProvider(model: string, apiKey: string) {
  // Similar approach for Google
  return createOpenAI({
    apiKey,
  })(model);
}

export async function POST(req: Request) {
  try {
    const { messages, provider, model, apiKey, ollamaEndpoint } = await req.json();

    if (!messages || !Array.isArray(messages)) {
      return new Response('Invalid messages format', { status: 400 });
    }

    let selectedModel;
    const effectiveModel = model || 'llama3.2';
    
    switch (provider) {
      case 'ollama':
        selectedModel = getOllamaProvider(effectiveModel, ollamaEndpoint || 'http://localhost:11434');
        break;
      case 'openai':
        if (!apiKey) {
          return new Response('API key required for OpenAI', { status: 400 });
        }
        selectedModel = getOpenAIProvider(effectiveModel || 'gpt-4', apiKey);
        break;
      case 'anthropic':
        if (!apiKey) {
          return new Response('API key required for Anthropic', { status: 400 });
        }
        selectedModel = getAnthropicCompatibleProvider(effectiveModel || 'claude-3-sonnet-20240229', apiKey);
        break;
      case 'google':
        if (!apiKey) {
          return new Response('API key required for Google', { status: 400 });
        }
        selectedModel = getGoogleCompatibleProvider(effectiveModel || 'gemini-pro', apiKey);
        break;
      default:
        // Default to Ollama
        selectedModel = getOllamaProvider(effectiveModel);
    }

    const result = await streamText({
      model: selectedModel,
      messages,
      system: `You are Senter, a thoughtful, intelligent, and helpful AI assistant. You are running locally on the user's Mac and have access to their context (clipboard, recent files, etc.).

Your personality:
- Be concise but thorough when needed
- Be friendly and conversational
- Help with coding, writing, analysis, and creative tasks
- When you don't know something, say so honestly
- Format code with proper syntax highlighting using markdown code blocks`,
    });

    return result.toDataStreamResponse();
  } catch (error) {
    console.error('Chat API error:', error);
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    
    // Check for common Ollama errors
    if (errorMessage.includes('ECONNREFUSED') || errorMessage.includes('fetch failed')) {
      return new Response(
        JSON.stringify({ 
          error: 'Cannot connect to Ollama. Make sure Ollama is running (ollama serve).' 
        }),
        { status: 503, headers: { 'Content-Type': 'application/json' } }
      );
    }
    
    return new Response(
      JSON.stringify({ error: errorMessage }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

