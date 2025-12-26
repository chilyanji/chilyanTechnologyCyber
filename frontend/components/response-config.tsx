"use client"

import { useState } from "react"

interface ResponsePolicy {
  threat_level: string
  auto_quarantine: boolean
  auto_block: boolean
  send_alert: boolean
  require_approval: boolean
}

export default function ResponseConfig() {
  const [policies, setPolicies] = useState<ResponsePolicy[]>([
    {
      threat_level: "malicious",
      auto_quarantine: true,
      auto_block: true,
      send_alert: true,
      require_approval: false,
    },
    {
      threat_level: "suspicious",
      auto_quarantine: true,
      auto_block: false,
      send_alert: true,
      require_approval: false,
    },
    {
      threat_level: "safe",
      auto_quarantine: false,
      auto_block: false,
      send_alert: false,
      require_approval: false,
    },
  ])

  const togglePolicy = (index: number, key: keyof ResponsePolicy) => {
    const updated = [...policies]
    if (key !== "threat_level") {
      updated[index] = {
        ...updated[index],
        [key]: !updated[index][key],
      }
      setPolicies(updated)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Response Policies</h1>
        <p className="text-border">Configure automated threat response actions</p>
      </div>

      <div className="grid gap-4">
        {policies.map((policy, idx) => (
          <div key={policy.threat_level} className="bg-card border border-border rounded-xl p-6">
            <h3 className="text-lg font-bold text-foreground mb-4 capitalize">{policy.threat_level} Threats</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={policy.auto_quarantine}
                  onChange={() => togglePolicy(idx, "auto_quarantine")}
                  className="w-5 h-5 rounded border-border"
                />
                <span className="text-foreground">Auto Quarantine</span>
              </label>

              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={policy.auto_block}
                  onChange={() => togglePolicy(idx, "auto_block")}
                  className="w-5 h-5 rounded border-border"
                />
                <span className="text-foreground">Auto Block</span>
              </label>

              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={policy.send_alert}
                  onChange={() => togglePolicy(idx, "send_alert")}
                  className="w-5 h-5 rounded border-border"
                />
                <span className="text-foreground">Send Alert</span>
              </label>

              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={policy.require_approval}
                  onChange={() => togglePolicy(idx, "require_approval")}
                  className="w-5 h-5 rounded border-border"
                />
                <span className="text-foreground">Require Approval</span>
              </label>
            </div>
          </div>
        ))}
      </div>

      <button className="w-full bg-gradient-to-r from-primary to-primary-dark text-white font-semibold py-3 rounded-lg hover:opacity-90 transition">
        Save Policies
      </button>
    </div>
  )
}
