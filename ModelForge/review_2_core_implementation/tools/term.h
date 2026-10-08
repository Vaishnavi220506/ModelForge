#pragma once

// Small ANSI terminal toolkit for the ModelForge studio and benchmark tools.
// Works in the VS Code integrated terminal, Windows Terminal, PowerShell and
// any xterm-compatible terminal. Use --ascii / --no-color (or NO_COLOR) on
// terminals without Unicode or colour support.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace term {

struct Settings {
    bool color = true;
    bool unicode = true;
};

inline Settings& settings() {
    static Settings value;
    return value;
}

inline void init(bool forceAscii = false, bool forceNoColor = false) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (output != INVALID_HANDLE_VALUE && GetConsoleMode(output, &mode)) {
        SetConsoleMode(output, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#endif
    settings().color = !forceNoColor && std::getenv("NO_COLOR") == nullptr;
    settings().unicode = !forceAscii;
}

// ------------------------------------------------------------------ colours

inline std::string rgb(int r, int g, int b) {
    if (!settings().color) return {};
    return "\x1b[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) +
           "m";
}
inline std::string bg(int r, int g, int b) {
    if (!settings().color) return {};
    return "\x1b[48;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) +
           "m";
}
inline std::string code(const char* sequence) {
    return settings().color ? std::string(sequence) : std::string();
}
inline std::string reset() { return code("\x1b[0m"); }
inline std::string bold() { return code("\x1b[1m"); }
inline std::string dim() { return code("\x1b[2m"); }

// Palette: a calm indigo/teal scheme that reads well on dark VS Code themes.
inline std::string accent() { return rgb(129, 140, 248); }   // indigo
inline std::string teal() { return rgb(45, 212, 191); }
inline std::string good() { return rgb(74, 222, 128); }
inline std::string warn() { return rgb(251, 191, 36); }
inline std::string bad() { return rgb(248, 113, 113); }
inline std::string muted() { return rgb(148, 163, 184); }
inline std::string text() { return rgb(226, 232, 240); }
inline std::string pink() { return rgb(244, 114, 182); }

inline std::string paint(const std::string& colour, const std::string& value) {
    return colour + value + reset();
}

// ------------------------------------------------------------------ glyphs

struct Glyphs {
    const char* tl; const char* tr; const char* bl; const char* br;
    const char* h; const char* v; const char* lt; const char* rt;
    const char* check; const char* cross; const char* arrow; const char* dot;
    const char* down;
};

inline const Glyphs& glyphs() {
    static const Glyphs unicode{"╭", "╮", "╰", "╯", "─", "│", "├", "┤",
                                "✔", "✘", "→", "●", "▼"};
    static const Glyphs ascii{"+", "+", "+", "+", "-", "|", "+", "+",
                              "OK", "X", "->", "*", "v"};
    return settings().unicode ? unicode : ascii;
}

// Visible width: ignores ANSI escapes and counts UTF-8 code points.
inline std::size_t visibleWidth(const std::string& value) {
    std::size_t width = 0;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (c == 0x1b) {
            while (i < value.size() && value[i] != 'm') ++i;
            continue;
        }
        if ((c & 0xC0) != 0x80) ++width;
    }
    return width;
}

inline std::string repeat(const std::string& unit, std::size_t count) {
    std::string result;
    for (std::size_t i = 0; i < count; ++i) result += unit;
    return result;
}

inline std::string padRight(const std::string& value, std::size_t width) {
    const std::size_t visible = visibleWidth(value);
    return visible >= width ? value : value + std::string(width - visible, ' ');
}

inline std::string padLeft(const std::string& value, std::size_t width) {
    const std::size_t visible = visibleWidth(value);
    return visible >= width ? value : std::string(width - visible, ' ') + value;
}

inline std::string fixed(double value, int digits = 2) {
    std::ostringstream output;
    output.setf(std::ios::fixed);
    output.precision(digits);
    output << value;
    return output.str();
}

inline std::string percent(double ratio, int digits = 1) { return fixed(ratio * 100.0, digits) + "%"; }

// ------------------------------------------------------------------ widgets

// Horizontal bar with 1/8 block resolution.
inline std::string bar(double ratio, std::size_t width, const std::string& colour = teal()) {
    ratio = std::clamp(ratio, 0.0, 1.0);
    if (!settings().unicode) {
        const std::size_t filled = static_cast<std::size_t>(std::round(ratio * width));
        return colour + std::string(filled, '#') + reset() + muted() +
               std::string(width - filled, '.') + reset();
    }
    static const char* eighths[] = {"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"};
    const double cells = ratio * static_cast<double>(width);
    std::size_t full = static_cast<std::size_t>(cells);
    std::size_t part = static_cast<std::size_t>((cells - static_cast<double>(full)) * 8.0);
    std::string result = colour + repeat("█", full);
    std::size_t used = full;
    if (part > 0 && used < width) {
        result += eighths[part];
        ++used;
    }
    result += reset() + rgb(51, 65, 85) + repeat("·", width - used) + reset();
    return result;
}

inline std::string sparkline(const std::vector<double>& values, const std::string& colour = teal()) {
    if (values.empty()) return {};
    static const char* blocks[] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    const double low = *std::min_element(values.begin(), values.end());
    const double high = *std::max_element(values.begin(), values.end());
    std::string result = colour;
    for (const double value : values) {
        const double t = high - low < 1e-12 ? 1.0 : (value - low) / (high - low);
        const int index = std::clamp(static_cast<int>(std::round(t * 7.0)), 0, 7);
        result += settings().unicode ? blocks[index] : (t > 0.5 ? "#" : ".");
    }
    return result + reset();
}

inline void box(const std::string& title, const std::vector<std::string>& lines,
                std::size_t width = 78, const std::string& colour = accent()) {
    const Glyphs& g = glyphs();
    std::size_t inner = width - 2;
    std::string top = colour + g.tl + g.h + reset();
    std::string label = title.empty() ? "" : " " + bold() + text() + title + reset() + " ";
    top += label + colour;
    const std::size_t used = 1 + visibleWidth(label);
    top += repeat(g.h, inner > used ? inner - used : 0) + g.tr + reset();
    std::cout << top << "\n";
    for (const std::string& line : lines) {
        std::cout << colour << g.v << reset() << " " << padRight(line, inner - 2) << " " << colour
                  << g.v << reset() << "\n";
    }
    std::cout << colour << g.bl << repeat(g.h, inner) << g.br << reset() << "\n";
}

inline void rule(const std::string& title, std::size_t width = 78) {
    const Glyphs& g = glyphs();
    const std::string label = " " + bold() + accent() + title + reset() + " ";
    const std::size_t used = visibleWidth(label) + 2;
    std::cout << "\n" << muted() << g.h << g.h << reset() << label << muted()
              << repeat(g.h, width > used ? width - used : 0) << reset() << "\n\n";
}

struct Table {
    std::vector<std::string> headers;
    std::vector<bool> rightAlign;
    std::vector<std::vector<std::string>> rows;

    void print(std::ostream& output = std::cout) const {
        std::vector<std::size_t> widths(headers.size(), 0);
        for (std::size_t c = 0; c < headers.size(); ++c) widths[c] = visibleWidth(headers[c]);
        for (const auto& row : rows) {
            for (std::size_t c = 0; c < row.size() && c < widths.size(); ++c) {
                widths[c] = std::max(widths[c], visibleWidth(row[c]));
            }
        }
        const auto cell = [&](const std::string& value, std::size_t c) {
            const bool right = c < rightAlign.size() && rightAlign[c];
            return right ? padLeft(value, widths[c]) : padRight(value, widths[c]);
        };
        output << "  ";
        for (std::size_t c = 0; c < headers.size(); ++c) {
            output << bold() << muted() << cell(headers[c], c) << reset() << "  ";
        }
        output << "\n  ";
        for (std::size_t c = 0; c < headers.size(); ++c) {
            output << rgb(51, 65, 85) << repeat(glyphs().h, widths[c]) << reset() << "  ";
        }
        output << "\n";
        for (const auto& row : rows) {
            output << "  ";
            for (std::size_t c = 0; c < headers.size(); ++c) {
                output << cell(c < row.size() ? row[c] : "", c) << "  ";
            }
            output << "\n";
        }
    }
};

inline void progress(std::size_t done, std::size_t total, const std::string& label,
                     std::size_t width = 36) {
    const double ratio = total == 0 ? 1.0 : static_cast<double>(done) / static_cast<double>(total);
    std::cout << "\r  " << bar(ratio, width, accent()) << " " << padLeft(percent(ratio, 0), 4)
              << "  " << muted() << padRight(label, 24) << reset() << std::flush;
    if (done >= total) std::cout << "\n";
}

inline void banner() {
    const std::vector<std::string> art = {
        "  __  __           _      _ _____                    ",
        " |  \\/  | ___   __| | ___| |  ___|__  _ __ __ _  ___ ",
        " | |\\/| |/ _ \\ / _` |/ _ \\ |  |_ / _ \\| '__/ _` |/ _ \\",
        " | |  | | (_) | (_| |  __/ |  _| (_) | | | (_| |  __/",
        " |_|  |_|\\___/ \\__,_|\\___|_|_|  \\___/|_|  \\__, |\\___|",
        "                                          |___/      "};
    const int colours[][3] = {{165, 180, 252}, {129, 140, 248}, {99, 102, 241},
                              {45, 212, 191},  {20, 184, 166},  {13, 148, 136}};
    std::cout << "\n";
    for (std::size_t i = 0; i < art.size(); ++i) {
        std::cout << "  " << rgb(colours[i][0], colours[i][1], colours[i][2]) << bold() << art[i]
                  << reset() << "\n";
    }
    std::cout << "  " << muted() << "by " << reset() << bold() << pink() << "Vaishnavi" << reset()
              << "\n";
}

}  // namespace term
