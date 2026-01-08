"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Server, Key, Cpu, Check, AlertCircle, RefreshCw } from "lucide-react";
import { useStore, type AIProvider } from "@/lib/store";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const providers: { id: AIProvider; name: string; description: string; requiresKey: boolean }[] = [
  { id: 'ollama', name: 'Ollama (Local)', description: 'Run AI models locally on your Mac', requiresKey: false },
  { id: 'openai', name: 'OpenAI', description: 'GPT-4, GPT-3.5, etc.', requiresKey: true },
  { id: 'anthropic', name: 'Anthropic', description: 'Claude 3 models', requiresKey: true },
  { id: 'google', name: 'Google', description: 'Gemini models', requiresKey: true },
];

const defaultModels: Record<AIProvider, string[]> = {
  ollama: ['llama3.2', 'llama3.1', 'mistral', 'codellama', 'phi3', 'gemma2'],
  openai: ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
  anthropic: ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
  google: ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro'],
};

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const { settings, updateSettings, ollamaConnected, setOllamaConnected } = useStore();
  const [localApiKey, setLocalApiKey] = useState(settings.apiKey || '');
  const [localModel, setLocalModel] = useState(settings.model);
  const [localEndpoint, setLocalEndpoint] = useState(settings.ollamaEndpoint);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isCheckingConnection, setIsCheckingConnection] = useState(false);

  // Check Ollama connection
  const checkOllamaConnection = async () => {
    setIsCheckingConnection(true);
    try {
      if (window.electronAPI) {
        const result = await window.electronAPI.checkOllama();
        setOllamaConnected(result.running);
        if (result.models) {
          setAvailableModels(result.models.map((m: { name: string }) => m.name));
          updateSettings({ ollamaModels: result.models.map((m: { name: string }) => m.name) });
        }
      } else {
        // Fallback for browser testing
        const response = await fetch(`${localEndpoint}/api/tags`);
        if (response.ok) {
          const data = await response.json();
          setOllamaConnected(true);
          const models = data.models?.map((m: { name: string }) => m.name) || [];
          setAvailableModels(models);
          updateSettings({ ollamaModels: models });
        } else {
          setOllamaConnected(false);
        }
      }
    } catch {
      setOllamaConnected(false);
      setAvailableModels([]);
    }
    setIsCheckingConnection(false);
  };

  useEffect(() => {
    if (isOpen && settings.provider === 'ollama') {
      checkOllamaConnection();
    }
  }, [isOpen, settings.provider]);

  useEffect(() => {
    setLocalApiKey(settings.apiKey || '');
    setLocalModel(settings.model);
    setLocalEndpoint(settings.ollamaEndpoint);
  }, [settings]);

  const handleSave = () => {
    updateSettings({
      apiKey: localApiKey || undefined,
      model: localModel,
      ollamaEndpoint: localEndpoint,
    });
    onClose();
  };

  const handleProviderChange = (provider: AIProvider) => {
    updateSettings({ provider });
    // Set default model for the provider
    const defaultModel = defaultModels[provider][0];
    setLocalModel(defaultModel);
    updateSettings({ model: defaultModel });
  };

  const currentModels = settings.provider === 'ollama' 
    ? (availableModels.length > 0 ? availableModels : defaultModels.ollama)
    : defaultModels[settings.provider];

  const requiresKey = providers.find(p => p.id === settings.provider)?.requiresKey ?? false;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="relative w-full max-w-lg glass-panel overflow-hidden" data-ui-panel>
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-brand-light/10">
                <h2 className="text-xl font-medium text-brand-light">Settings</h2>
                <motion.button
                  onClick={onClose}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  className="w-8 h-8 rounded-lg flex items-center justify-center bg-brand-light/10 hover:bg-brand-light/20 transition-colors"
                >
                  <X className="w-4 h-4 text-brand-light/70" />
                </motion.button>
              </div>

              {/* Content */}
              <div className="p-6 space-y-6 max-h-[60vh] overflow-y-auto">
                {/* Provider Selection */}
                <div className="space-y-3">
                  <label className="flex items-center gap-2 text-sm font-medium text-brand-light/80">
                    <Server className="w-4 h-4" />
                    AI Provider
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    {providers.map((provider) => (
                      <motion.button
                        key={provider.id}
                        onClick={() => handleProviderChange(provider.id)}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className={`p-3 rounded-xl text-left transition-all ${
                          settings.provider === provider.id
                            ? 'bg-brand-primary/20 border-2 border-brand-primary/40'
                            : 'bg-brand-light/5 border-2 border-transparent hover:bg-brand-light/10'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-brand-light">
                            {provider.name}
                          </span>
                          {settings.provider === provider.id && (
                            <Check className="w-4 h-4 text-brand-accent" />
                          )}
                        </div>
                        <span className="text-xs text-brand-light/50">{provider.description}</span>
                      </motion.button>
                    ))}
                  </div>
                </div>

                {/* Ollama Status */}
                {settings.provider === 'ollama' && (
                  <div className="flex items-center justify-between p-3 rounded-xl bg-brand-light/5 border border-brand-light/10">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${ollamaConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                      <span className="text-sm text-brand-light/70">
                        {ollamaConnected ? 'Ollama Connected' : 'Ollama Not Running'}
                      </span>
                    </div>
                    <motion.button
                      onClick={checkOllamaConnection}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      disabled={isCheckingConnection}
                      className="p-2 rounded-lg bg-brand-light/10 hover:bg-brand-light/20 transition-colors disabled:opacity-50"
                    >
                      <RefreshCw className={`w-4 h-4 text-brand-light/70 ${isCheckingConnection ? 'animate-spin' : ''}`} />
                    </motion.button>
                  </div>
                )}

                {/* Ollama Endpoint */}
                {settings.provider === 'ollama' && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-brand-light/80">Ollama Endpoint</label>
                    <input
                      type="text"
                      value={localEndpoint}
                      onChange={(e) => setLocalEndpoint(e.target.value)}
                      placeholder="http://localhost:11434"
                      className="w-full px-4 py-3 rounded-xl bg-brand-light/5 border border-brand-light/10 text-brand-light placeholder-brand-light/30 outline-none focus:border-brand-primary/50 transition-colors"
                    />
                  </div>
                )}

                {/* API Key (for cloud providers) */}
                {requiresKey && (
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 text-sm font-medium text-brand-light/80">
                      <Key className="w-4 h-4" />
                      API Key
                    </label>
                    <input
                      type="password"
                      value={localApiKey}
                      onChange={(e) => setLocalApiKey(e.target.value)}
                      placeholder={`Enter your ${providers.find(p => p.id === settings.provider)?.name} API key`}
                      className="w-full px-4 py-3 rounded-xl bg-brand-light/5 border border-brand-light/10 text-brand-light placeholder-brand-light/30 outline-none focus:border-brand-primary/50 transition-colors"
                    />
                    {!localApiKey && (
                      <div className="flex items-center gap-2 text-xs text-yellow-500/80">
                        <AlertCircle className="w-3 h-3" />
                        API key is required for this provider
                      </div>
                    )}
                  </div>
                )}

                {/* Model Selection */}
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm font-medium text-brand-light/80">
                    <Cpu className="w-4 h-4" />
                    Model
                  </label>
                  <select
                    value={localModel}
                    onChange={(e) => setLocalModel(e.target.value)}
                    className="w-full px-4 py-3 rounded-xl bg-brand-light/5 border border-brand-light/10 text-brand-light outline-none focus:border-brand-primary/50 transition-colors appearance-none cursor-pointer"
                  >
                    {currentModels.map((model) => (
                      <option key={model} value={model} className="bg-brand-dark text-brand-light">
                        {model}
                      </option>
                    ))}
                  </select>
                  {settings.provider === 'ollama' && availableModels.length === 0 && (
                    <p className="text-xs text-brand-light/50">
                      No models detected. Run <code className="px-1 py-0.5 bg-brand-light/10 rounded">ollama pull llama3.2</code> to download a model.
                    </p>
                  )}
                </div>
              </div>

              {/* Footer */}
              <div className="flex items-center justify-end gap-3 p-6 border-t border-brand-light/10">
                <motion.button
                  onClick={onClose}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="px-4 py-2 rounded-xl text-sm font-medium text-brand-light/70 hover:text-brand-light hover:bg-brand-light/10 transition-colors"
                >
                  Cancel
                </motion.button>
                <motion.button
                  onClick={handleSave}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="px-4 py-2 rounded-xl text-sm font-medium bg-brand-accent text-brand-light hover:bg-brand-accent/90 transition-colors"
                >
                  Save Changes
                </motion.button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

