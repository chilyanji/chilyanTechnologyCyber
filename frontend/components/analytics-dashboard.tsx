"use client"

import { useState } from "react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"

interface AnalyticsData {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
}

interface ThreatIndicator {
  indicator: string
  count: number
  percentage: number
}

export default function AnalyticsDashboard() {
  const [performance, setPerformance] = useState<AnalyticsData>({
    accuracy: 94.2,
    precision: 92.8,
    recall: 89.5,
    f1_score: 91.1,
  })

  const [topIndicators, setTopIndicators] = useState<ThreatIndicator[]>([
    { indicator: "Suspicious Domain", count: 45, percentage: 27.8 },
    { indicator: "Homograph Attack", count: 38, percentage: 23.5 },
    { indicator: "Suspicious Content", count: 32, percentage: 19.8 },
    { indicator: "Urgent Language", count: 28, percentage: 17.3 },
    { indicator: "Shortened URL", count: 19, percentage: 11.7 },
  ])

  const pieData = [
    { name: "Suspicious Domain", value: 27.8 },
    { name: "Homograph Attack", value: 23.5 },
    { name: "Suspicious Content", value: 19.8 },
    { name: "Urgent Language", value: 17.3 },
    { name: "Shortened URL", value: 11.6 },
  ]

  const COLORS = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6"]

  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Analytics & Insights</h1>
        <p className="text-border">System performance metrics and threat intelligence</p>
      </div>

      {/* Model Performance */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-xl p-6">
          <p className="text-foreground/70 text-sm mb-2">Accuracy</p>
          <p className="text-3xl font-bold text-primary">{performance.accuracy}%</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-6">
          <p className="text-foreground/70 text-sm mb-2">Precision</p>
          <p className="text-3xl font-bold text-safe">{performance.precision}%</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-6">
          <p className="text-foreground/70 text-sm mb-2">Recall</p>
          <p className="text-3xl font-bold text-suspicious">{performance.recall}%</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-6">
          <p className="text-foreground/70 text-sm mb-2">F1 Score</p>
          <p className="text-3xl font-bold text-primary">{performance.f1_score}%</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Threat Indicators */}
        <div className="bg-card border border-border rounded-xl p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">Top Threat Indicators</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topIndicators}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3547" />
              <XAxis dataKey="indicator" angle={-45} textAnchor="end" height={80} stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1f3a",
                  border: "1px solid #2d3547",
                  borderRadius: "8px",
                }}
              />
              <Bar dataKey="count" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Threat Distribution Pie */}
        <div className="bg-card border border-border rounded-xl p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">Threat Indicators Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={2} dataKey="value">
                {COLORS.map((color, index) => (
                  <Cell key={`cell-${index}`} fill={color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1f3a",
                  border: "1px solid #2d3547",
                  borderRadius: "8px",
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Threat Indicators Table */}
      <div className="bg-card border border-border rounded-xl p-6">
        <h3 className="text-lg font-bold text-foreground mb-4">Detailed Threat Analysis</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-foreground/70 font-medium">Threat Indicator</th>
                <th className="text-left py-3 px-4 text-foreground/70 font-medium">Count</th>
                <th className="text-left py-3 px-4 text-foreground/70 font-medium">Percentage</th>
                <th className="text-left py-3 px-4 text-foreground/70 font-medium">Trend</th>
              </tr>
            </thead>
            <tbody>
              {topIndicators.map((indicator, idx) => (
                <tr key={idx} className="border-b border-border hover:bg-border/30 transition">
                  <td className="py-3 px-4 text-foreground">{indicator.indicator}</td>
                  <td className="py-3 px-4 text-foreground font-semibold">{indicator.count}</td>
                  <td className="py-3 px-4 text-foreground">{indicator.percentage}%</td>
                  <td className="py-3 px-4">
                    <span className="text-malicious text-sm">↑ +{(indicator.percentage * 0.15).toFixed(1)}%</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
