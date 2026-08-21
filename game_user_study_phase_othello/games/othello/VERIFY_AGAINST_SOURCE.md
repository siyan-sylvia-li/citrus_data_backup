# Checking our solutions against othelloclub's own answers

Their board page is behind a Google reCAPTCHA challenge, so this cannot be
scripted — it has to be a person clicking through in a normal browser. Each
row below is one puzzle: open the link, play the move in **our answer**, and
the site should tell you that you won.

Notes on reading this table:

- `their side` is who moves in the site's own version. For `White`, we
  colour-flipped on import so the participant is always Black; the SQUARE is
  unchanged, so play our answer as White there.
- `margin` is our exact predicted final disc difference for the side to move.
  The site only judges win/lose, so a match confirms the sign, not the margin.
- Every other legal move loses in all 20 puzzles, so playing one of the
  `alternatives` is an equally valid check in the opposite direction: the site
  should say you lost.

| date | their side | our answer | margin | alternatives (all lose) | link |
|---|---|---|---|---|---|
| 20260729 | Black | **h4** | +2 | f1 (-2), b7 (-12), h7 (-16), h2 (-18), g2 (-28) | [open](https://othelloclub.com/en/board.php?board=2222200022122200212121222112212021212121222112212012212001111110&turn=1&date=20260729) |
| 20260728 | White | **h4** | +14 | a2 (-4), a7 (-4), h7 (-4), h2 (-6), g2 (-8) | [open](https://othelloclub.com/en/board.php?board=1111111001221100111221111111121011211120111211120021211011111111&turn=2&date=20260728) |
| 20260727 | Black | **b7** | +18 | b6 (-14), g2 (-14), a7 (-18), c8 (-22), b8 (-26) | [open](https://othelloclub.com/en/board.php?board=0222222021221201011222211121222111221221201221210022222100021221&turn=1&date=20260727) |
| 20260726 | White | **a3** | +2 | a2 (-2), b7 (-4), a6 (-6), b1 (-6), a4 (-8), b2 (-8) | [open](https://othelloclub.com/en/board.php?board=0011112100121121011111210112112111222221011222212012222201111111&turn=2&date=20260726) |
| 20260725 | Black | **h1** | +4 | b7 (-4), a7 (-6), g1 (-10), b1 (-12), a2 (-18), h2 (-18), b2 (-18) | [open](https://othelloclub.com/en/board.php?board=0022220000212220222222222222122222111122221222220021122202222222&turn=1&date=20260725) |
| 20260724 | White | **a6** | +2 | h7 (-12), b7 (-16), b2 (-18), g7 (-23) | [open](https://othelloclub.com/en/board.php?board=0111111120112211221221212122112102111121012112111012210001111110&turn=2&date=20260724) |
| 20260723 | Black | **h8** | +5 | d8 (-2), b2 (-2), g8 (-5), f8 (-10), c2 (-14), d2 (-14), e8 (-16) | [open](https://othelloclub.com/en/board.php?board=0022222220001221222222212122212221222122212112222222222222200000&turn=1&date=20260723) |
| 20260722 | White | **g7** | +2 | b7 (-2), a6 (-4), a7 (-4), f7 (-10) | [open](https://othelloclub.com/en/board.php?board=1222222111211221221211211222211111212111011111110011200202222000&turn=2&date=20260722) |
| 20260721 | Black | **h3** | +4 | a8 (-2), g7 (-2), b1 (-8), b2 (-10), b8 (-24) | [open](https://othelloclub.com/en/board.php?board=0022220020222111221211102122112222221212222212222222210200111110&turn=1&date=20260721) |
| 20260720 | White | **h5** | +2 | h6 (-2), g8 (-2), h4 (-4), g7 (-4), h2 (-8), h3 (-10), g2 (-10) | [open](https://othelloclub.com/en/board.php?board=1111111112121100112121101111121012222120122212101111210122222200&turn=2&date=20260720) |
| 20260719 | Black | **a7** | +1 | c1 (-1), b2 (-2), c2 (-3), a6 (-8), a2 (-15), h7 (-18) | [open](https://othelloclub.com/en/board.php?board=0002221200022112122212221222222222122222021112220211112011111110&turn=1&date=20260719) |
| 20260718 | White | **a3** | +2 | b7 (-2), a6 (-4), b6 (-10), g1 (-22), b8 (-38) | [open](https://othelloclub.com/en/board.php?board=1211110011211112011211122111111211111112001111220011222200111121&turn=2&date=20260718) |
| 20260717 | Black | **h5** | +10 | h7 (-2), g2 (-2), g1 (-4), g7 (-4), g8 (-10), b7 (-16) | [open](https://othelloclub.com/en/board.php?board=2222220022122201211122112111122122112220222222222022220002222200&turn=1&date=20260717) |
| 20260716 | White | **d8** | +12 | g2 (-3), g1 (-10), b8 (-12), b2 (-16), c8 (-34) | [open](https://othelloclub.com/en/board.php?board=0111110020121101211121212112122121211121211121212111122100001121&turn=2&date=20260716) |
| 20260715 | Black | **h1** | +6 | g2 (-12), a2 (-16), a4 (-22), b6 (-24), h2 (-36) | [open](https://othelloclub.com/en/board.php?board=0111112000111200222122220212212222222212102212220022222211111112&turn=1&date=20260715) |
| 20260714 | White | **a7** | +4 | b2 (-6), a2 (-12), d8 (-18), b8 (-18), b7 (-22) | [open](https://othelloclub.com/en/board.php?board=0111111100112211111122111111121112111111112221110011121100102220&turn=2&date=20260714) |
| 20260713 | Black | **g7** | +8 | f8 (-4), b8 (-6), h7 (-8), b7 (-14), a7 (-16), b6 (-30) | [open](https://othelloclub.com/en/board.php?board=2222221211211112212211122112112222211222202212220022210000111000&turn=1&date=20260713) |
| 20260712 | White | **e1** | +10 | g1 (-18), b8 (-18), b7 (-18), g7 (-18), g2 (-28) | [open](https://othelloclub.com/en/board.php?board=1211020012211101121211111211222112111221122212211011210100111110&turn=2&date=20260712) |
| 20260711 | Black | **a1** | +6 | g7 (-2), g8 (-18), g2 (-18), b1 (-32), b8 (-32) | [open](https://othelloclub.com/en/board.php?board=0021110022221102212222222121212222221212212221220222220220222200&turn=1&date=20260711) |
| 20260710 | White | **a8** | +10 | h5 (-2), g3 (-8), b2 (-18), b7 (-28), a7 (-46) | [open](https://othelloclub.com/en/board.php?board=0111111010122220112112011221121111121110112111120011111101111121&turn=2&date=20260710) |

Generated by `python verify_against_source.py` (this file is its output).

## Why this is a weaker check than an engine diff

The site's task is "win from here", so its verdict is win/lose. Our claim is
the exact final disc margin under optimal play by both sides, which is
strictly stronger — the site can falsify our sign but never confirm our
margin. It also plays its own defence, which need not be the optimal defence
our tables assume, so a line can diverge from ours and still end in a win.
For real external agreement, diff against an independent solver (Edax).
