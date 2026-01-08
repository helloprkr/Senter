/**
 * Senter API Client
 * Connects to the local Senter HTTP server
 */

const SENTER_API_BASE = 'http://127.0.0.1:8420/api';

export interface SenterGoal {
  id: string;
  description: string;
  progress: number;
  category: string;
  status: string;
  source?: string;
  confidence?: number;
}

export interface SenterSuggestion {
  type: string;
  content: string;
  priority?: number;
  reason?: string;
}

export interface SenterActivity {
  context: string;
  project: string | null;
  duration_minutes: number;
  snapshot_count: number;
}

export interface SenterStatus {
  status: string;
  trust_level: number;
  memory_count: number;
  goal_count: number;
  uptime_hours: number;
}

export interface SenterAwayInfo {
  summary: string;
  research: Array<{
    goal: string;
    summary: string;
    sources: string[];
  }>;
  insights: string[];
  duration_hours: number;
}

export interface SenterChatResponse {
  response: string;
  suggestions: SenterSuggestion[];
  goals: SenterGoal[];
  fitness: number;
  mode: string;
  trust_level: number;
  episode_id?: string;
}

export interface SenterMemory {
  input: string;
  response: string;
  timestamp: string | null;
  fitness: number | null;
}

class SenterAPI {
  private baseUrl: string;

  constructor(baseUrl: string = SENTER_API_BASE) {
    this.baseUrl = baseUrl;
  }

  async chat(message: string): Promise<SenterChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Chat request failed');
    }

    return response.json();
  }

  async getGoals(): Promise<SenterGoal[]> {
    const response = await fetch(`${this.baseUrl}/goals`);
    if (!response.ok) return [];
    const data = await response.json();
    return data.goals || [];
  }

  async getActivity(): Promise<SenterActivity> {
    const response = await fetch(`${this.baseUrl}/activity`);
    if (!response.ok) {
      return { context: 'unknown', project: null, duration_minutes: 0, snapshot_count: 0 };
    }
    return response.json();
  }

  async getSuggestions(): Promise<SenterSuggestion[]> {
    const response = await fetch(`${this.baseUrl}/suggestions`);
    if (!response.ok) return [];
    const data = await response.json();
    return data.suggestions || [];
  }

  async getAwayInfo(): Promise<SenterAwayInfo> {
    const response = await fetch(`${this.baseUrl}/away`);
    if (!response.ok) {
      return { summary: '', research: [], insights: [], duration_hours: 0 };
    }
    return response.json();
  }

  async getStatus(): Promise<SenterStatus> {
    const response = await fetch(`${this.baseUrl}/status`);
    if (!response.ok) {
      return { status: 'offline', trust_level: 0, memory_count: 0, goal_count: 0, uptime_hours: 0 };
    }
    return response.json();
  }

  async getMemories(): Promise<SenterMemory[]> {
    const response = await fetch(`${this.baseUrl}/memories`);
    if (!response.ok) return [];
    const data = await response.json();
    return data.memories || [];
  }

  async checkConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/status`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

export const senterAPI = new SenterAPI();
export default senterAPI;
