"use client"

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

const data = [
  { date: "Mon", safe: 450, suspicious: 80, malicious: 15 },
  { date: "Tue", safe: 520, suspicious: 95, malicious: 22 },
  { date: "Wed", safe: 480, suspicious: 72, malicious: 18 },
  { date: "Thu", safe: 610, suspicious: 110, malicious: 35 },
  { date: "Fri", safe: 680, suspicious: 140, malicious: 45 },
  { date: "Sat", safe: 520, suspicious: 85, malicious: 28 },
  { date: "Sun", safe: 450, suspicious: 70, malicious: 12 },
]

export default function ThreatChart() {
  return (
    <div className="bg-card border border-border rounded-xl p-6">
      <h3 className="text-lg font-bold text-foreground mb-6">Weekly Threat Trends</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2d3547" />
          <XAxis dataKey="date" stroke="#64748b" />
          <YAxis stroke="#64748b" />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1a1f3a",
              border: "1px solid #2d3547",
              borderRadius: "8px",
            }}
          />
          <Line type="monotone" dataKey="safe" stroke="#10b981" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="suspicious" stroke="#f59e0b" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="malicious" stroke="#ef4444" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
