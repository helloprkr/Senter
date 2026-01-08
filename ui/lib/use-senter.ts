"use client";

import { useEffect, useCallback, useRef } from 'react';
import { useStore } from './store';
import senterAPI from './senter-api';

export function useSenter() {
  const {
    setSenterConnected,
    setSenterGoals,
    setSenterSuggestions,
    setSenterActivity,
    setSenterStatus,
    setSenterAwayInfo,
    setShowAwayModal,
    senterConnected,
    senterGoals,
    senterSuggestions,
    senterActivity,
    senterStatus,
    senterAwayInfo,
    showAwayModal,
  } = useStore();

  const lastActiveRef = useRef<Date>(new Date());
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const checkConnection = useCallback(async () => {
    const connected = await senterAPI.checkConnection();
    setSenterConnected(connected);
    return connected;
  }, [setSenterConnected]);

  const fetchGoals = useCallback(async () => {
    const goals = await senterAPI.getGoals();
    setSenterGoals(goals);
  }, [setSenterGoals]);

  const fetchSuggestions = useCallback(async () => {
    const suggestions = await senterAPI.getSuggestions();
    setSenterSuggestions(suggestions);
  }, [setSenterSuggestions]);

  const fetchActivity = useCallback(async () => {
    const activity = await senterAPI.getActivity();
    setSenterActivity(activity);
  }, [setSenterActivity]);

  const fetchStatus = useCallback(async () => {
    const status = await senterAPI.getStatus();
    setSenterStatus(status);
  }, [setSenterStatus]);

  const checkAwayStatus = useCallback(async () => {
    const awayInfo = await senterAPI.getAwayInfo();
    if (awayInfo.duration_hours > 0.5 && awayInfo.summary) {
      setSenterAwayInfo(awayInfo);
      setShowAwayModal(true);
    }
  }, [setSenterAwayInfo, setShowAwayModal]);

  const refreshAll = useCallback(async () => {
    const connected = await checkConnection();
    if (connected) {
      await Promise.all([
        fetchGoals(),
        fetchSuggestions(),
        fetchActivity(),
        fetchStatus(),
      ]);
    }
  }, [checkConnection, fetchGoals, fetchSuggestions, fetchActivity, fetchStatus]);

  // Initial connection and polling
  useEffect(() => {
    refreshAll();
    checkAwayStatus();

    // Poll every 30 seconds
    pollIntervalRef.current = setInterval(() => {
      refreshAll();
    }, 30000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [refreshAll, checkAwayStatus]);

  // Track user activity for away detection
  useEffect(() => {
    const handleActivity = () => {
      lastActiveRef.current = new Date();
    };

    window.addEventListener('mousemove', handleActivity);
    window.addEventListener('keydown', handleActivity);

    return () => {
      window.removeEventListener('mousemove', handleActivity);
      window.removeEventListener('keydown', handleActivity);
    };
  }, []);

  return {
    connected: senterConnected,
    goals: senterGoals,
    suggestions: senterSuggestions,
    activity: senterActivity,
    status: senterStatus,
    awayInfo: senterAwayInfo,
    showAwayModal,
    setShowAwayModal,
    refreshAll,
    checkConnection,
  };
}

export default useSenter;
