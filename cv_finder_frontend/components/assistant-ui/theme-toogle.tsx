"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Moon, Sun } from "lucide-react";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";

type Theme = "light" | "dark";

const LOCAL_KEY = "theme";

export function ThemeToggle() {
  const [theme, setThemeState] = useState<Theme | null>(null);

  // Aplica la clase en el DOM
  const applyTheme = useCallback((t: Theme) => {
    if (t === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  // Lee localStorage o prefers-color-scheme al montar
  useEffect(() => {
    try {
      const stored = localStorage.getItem(LOCAL_KEY) as Theme | null;
      if (stored === "dark" || stored === "light") {
        setThemeState(stored);
        applyTheme(stored);
        return;
      }
    } catch {
      // localStorage puede fallar en entornos restrictivos
    }

    // fallback: preferencia del sistema
    const prefersDark =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;

    const initial: Theme = prefersDark ? "dark" : "light";
    setThemeState(initial);
    applyTheme(initial);
  }, [applyTheme]);

  // Sync entre pestañas
  useEffect(() => {
    function onStorage(e: StorageEvent) {
      if (e.key !== LOCAL_KEY) return;
      try {
        const newVal = e.newValue as Theme | null;
        if (newVal === "dark" || newVal === "light") {
          setThemeState(newVal);
          applyTheme(newVal);
        } else {
          // si se borra la key, reaplicar preferencia del sistema
          const prefersDark =
            window.matchMedia &&
            window.matchMedia("(prefers-color-scheme: dark)").matches;
          const fallback: Theme = prefersDark ? "dark" : "light";
          setThemeState(fallback);
          applyTheme(fallback);
        }
      } catch {}
    }
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [applyTheme]);

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(LOCAL_KEY, next);
    } catch {}
    setThemeState(next);
    applyTheme(next);
  };

  if (theme === null) {
    // evita render hasta que sabemos el tema
    return null;
  }

  return (
    <TooltipIconButton
      tooltip="Change theme"
      variant="ghost"
      size="icon"
      aria-pressed={theme === "dark"}
      onClick={toggle}
      className="ml-auto transition-all duration-300"
    >
      {theme === "dark" ? (
        <Moon className="size-5 rotate-0 scale-100 transition-all duration-300" />
      ) : (
        <Sun className="size-5 rotate-0 scale-100 transition-all duration-300" />
      )}
    </TooltipIconButton>
  );
}