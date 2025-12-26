"use client"

import { useState, useEffect } from "react"
import StatCard from "./stat-card"
import ThreatChart from "./threat-chart"
import RecentDetections from "./recent-detections"

export default function Dashboard() {
  const [stats, setStats] = useState({
    total_detections: 0,
    safe: 0,
    suspicious: 0,
    malicious: 0,
    safe_percentage: 0,
    malicious_percentage: 0,
  })

  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate fetching stats from API
    const mockStats = {
      total_detections: 2847,
      safe: 2198,
      suspicious: 487,
      malicious: 162,
      safe_percentage: 77.2,
      malicious_percentage: 5.7,
    }
    setStats(mockStats)
    setLoading(false)
  }, [])

  if (loading) {
    return <div className="text-center py-12">Loading dashboard...</div>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Security Overview</h1>
        <p className="text-border">Real-time threat monitoring and detection analytics</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Detections" value={stats.total_detections} icon="📊" color="primary" />
        <StatCard label="Safe" value={stats.safe} subtext={`${stats.safe_percentage}%`} icon="✓" color="safe" />
        <StatCard label="Suspicious" value={stats.suspicious} icon="⚠" color="suspicious" />
        <StatCard
          label="Malicious"
          value={stats.malicious}
          subtext={`${stats.malicious_percentage}% blocked`}
          icon="✕"
          color="malicious"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ThreatChart />
        </div>
        <div className="bg-card border border-border rounded-xl p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">Threat Distribution</h3>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-foreground/70">Safe</span>
                <span className="font-semibold text-foreground">{stats.safe_percentage}%</span>
              </div>
              <div className="w-full bg-border rounded-full h-2">
                <div className="bg-safe h-2 rounded-full" style={{ width: `${stats.safe_percentage}%` }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-foreground/70">Suspicious</span>
                <span className="font-semibold text-foreground">
                  {((stats.suspicious / stats.total_detections) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-border rounded-full h-2">
                <div
                  className="bg-suspicious h-2 rounded-full"
                  style={{
                    width: `${((stats.suspicious / stats.total_detections) * 100).toFixed(1)}%`,
                  }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-foreground/70">Malicious</span>
                <span className="font-semibold text-foreground">{stats.malicious_percentage}%</span>
              </div>
              <div className="w-full bg-border rounded-full h-2">
                <div className="bg-malicious h-2 rounded-full" style={{ width: `${stats.malicious_percentage}%` }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Detections */}
      <RecentDetections />
    </div>
  )
}
