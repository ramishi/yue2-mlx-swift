// CJK LyricTimelineTests.swift - reproduction for upstream issue #7
// Space-free CJK lyrics must still get sub-word timings.

import Foundation
import Testing
@testable import YuE2Core

private let cjkScore = """
X:1
T:
M:4/4
L:1/32
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
"c"c4d4e4f4|"G"g8e8|
V: Ins
Z4|
"""

@Suite("CJK lyric timing")
struct CJKLyricTimelineTests {
    /// RED: with whitespace tokenisation a whole line becomes one word, so the
    /// timeline only carries one syllable for a 13-character sentence. The real
    /// fix is a CJK-aware tokenizer in parseLyrics (see issue #7).
    @Test func spaceFreeCJKLineGetsSubWordTimings() throws {
        let timeline = try LyricTimeline.build(
            abc: cjkScore, lyrics: "[Verse]\n今天天氣很好我們一起去公園")
        let syllables = timeline.lines[0].words.flatMap(\.syllables)
        #expect(syllables.count > 1,
                "expected sub-word timings for space-free CJK, got \(syllables.count) syllable(s)")
    }

    /// GREEN: the same text with spaces already produces several words, which
    /// confirms the collapse is caused by the absence of whitespace, not by CJK.
    @Test func spacedCJKLineAlreadyGetsSeveralWords() throws {
        let timeline = try LyricTimeline.build(
            abc: cjkScore, lyrics: "[Verse]\n今天天氣很好  我們一起去公園")
        #expect(timeline.lines[0].words.count >= 2)
    }
}
