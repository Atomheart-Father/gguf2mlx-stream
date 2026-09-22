# Capability regression: Llama-3.2-1B-Instruct (source Q4_K_M vs converted MLX 4-bit)

- source: llama-completion 0.4.1, Llama-3.2-1B-Instruct-Q4_K_M.gguf, temp 0, max_tokens 512, --jinja, no thinking
- MLX: mlx-lm via gguf2mlx-stream llama conversion (4-bit, from local .integration-models fixture), temp 0, max_tokens 512, tokenizer chat template, no thinking
- protocol: identical chat template semantics (GGUF-embedded template vs the tokenizer files derived from it), single user turn with no system prompt on both sides, temp 0, max_tokens 512 (recorded in every run record and asserted identical across sides), thinking enabled (template default) on both sides, scored over the post-<think> text
- scoring gate: truncated / repetition-loop / empty / garbled outputs count as incorrect regardless of contained keywords; format checks are strict (single number, exactly three colors, one word, yes/no)
- note: Small-model protocol calibration pair. Scored with the strict gate (blocking anomalies count as incorrect; format checks strict; single number, exactly three colors, one word, yes/no).
- rescoring: 2026-09-22 — zh-cs-02 and en-cs-02 now accept equivalent correct answers (12个月/十二个月; 366 days); verdicts recomputed offline from the embedded per-item outputs, generation protocol and anomaly flags unchanged; changed verdicts: en-cs-02 (src False->True, mlx False->True); previous summary: src 42.5% / mlx 45.0%, verdict agreement 92.5%, flips ['zh-cs-09', 'en-cs-07', 'logic-01']

## Summary

| metric | source | MLX |
|---|---|---|
| gated accuracy (40 scored) | 45.0% | 47.5% |
| accuracy delta (MLX − source) | | +2.5% |
| anomaly rate (45 items) | 4.4% | 6.7% |
| total gen time (s) | 33.5 | 14.7 |

Answer agreement rate (surface form): **22.2%** · verdict agreement on scored items: **92.5%** (flips: zh-cs-09, en-cs-07, logic-01)

Anomaly breakdown (items affected, per type):

| type | source | MLX |
|---|---|---|
| repetition_loop | 1 | 2 |
| truncated | 2 | 2 |

| category | n | source | MLX |
|---|---|---|---|
| en_common | 10 | 8/10 | 9/10 |
| instruction | 5 | 1/5 | 1/5 |
| logic | 5 | 2/5 | 3/5 |
| math | 10 | 1/10 | 1/10 |
| zh_common | 10 | 6/10 | 5/10 |

## Per-question results

### zh-cs-01 — src ✓ / mlx ✓ / agree NO
- gold: 北京
- source (0.99s):
```
中国的首都是北京。
```
- mlx (0.15s):
```
中国的首都是北京市。
```

### zh-cs-02 — src ✓ / mlx ✓ / agree yes
- gold: 12
- source (0.49s):
```
一年有 12 个月。
```
- mlx (0.11s):
```
一年有 12 个月。
```

### zh-cs-03 — src ✓ / mlx ✓ / agree yes
- gold: H2O
- source (0.53s):
```
水的化学式为H2O。
```
- mlx (0.12s):
```
水的化学式为H2O。
```

### zh-cs-04 — src ✗ / mlx ✗ / agree NO
- gold: 汗滴禾下土
- source (0.63s):
```
“锄禾日当午，白日中升，天下人皆起，日下人皆行，人生无常，事事如常。”
```
- mlx (0.16s):
```
“我要去打击反对我和我兄弟的反对者。”
```

### zh-cs-05 — src ✓ / mlx ✓ / agree yes
- gold: 新疆
- source (0.53s):
```
中国面积最大的省级行政区是新疆。
```
- mlx (0.14s):
```
中国面积最大的省级行政区是新疆。
```

### zh-cs-06 — src ✗ / mlx ✗ / agree NO
- gold: 竹子
- source (0.56s):
```
大熊猫主要食物是鱼类、虾和其他小海洋动物。
```
- mlx (0.15s):
```
大熊猫的主要食物是鱼和其他海洋生 vật。
```

### zh-cs-07 — src ⚠repetition_loop,truncated / mlx ✗ / agree NO
- gold: 吴承恩
- source (3.18s):
```
《西游记》是中国古代最著名的传说故事之一，由《西游记》中的作者——《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的作者是《西游记》中的
```
- mlx (0.33s):
```
《西游记》是中国历史小说，由马超、马超长、马超长等四位同名作者合著。马超是中国最早的作家之一，他写下的《西游记》是中国历史小说中最早的作品之一。
```

### zh-cs-08 — src ✓ / mlx ✓ / agree NO
- gold: 东方
- source (0.71s):
```
太阳升起的方向取决于观察者的位置。太阳升起的方向通常被称为“东方”，而西方则被称为“西南”。
```
- mlx (0.12s):
```
太阳从东方升起。
```

### zh-cs-09 — src ✓ / mlx ✗ / agree NO
- gold: 珠穆朗玛峰
- source (0.64s):
```
世界上最高的山峰是珠穆朗玛峰。它位于尼泊尔的珠穆朗玛山群，高达8,848米。
```
- mlx (0.21s):
```
世界上最高的山峰是珠穆尔山。它的高度为7,421米，位于阿拉伯半島。
```

### zh-cs-10 — src ✗ / mlx ✗ / agree NO
- gold: 正月初一
- source (0.54s):
```
中国传统节日春节是农历的2月3日。
```
- mlx (0.15s):
```
中国传统节日春节是农历的第15天。
```

### en-cs-01 — src ✓ / mlx ✓ / agree yes
- gold: Paris
- source (0.49s):
```
The capital of France is Paris.
```
- mlx (0.12s):
```
The capital of France is Paris.
```

### en-cs-02 — src ✓ / mlx ✓ / agree NO
- gold: 366
- source (0.65s):
```
There are 366 days in a leap year. This extra day is added to the calendar to account for the Earth's orbit around the Sun, which takes approximately 365.24 days to complete.
```
- mlx (0.25s):
```
There are 366 days in a leap year. This is because a leap year has 366 days, which is one day more than a standard year has 365 days.
```

### en-cs-03 — src ✓ / mlx ✓ / agree yes
- gold: Au
- source (0.51s):
```
The chemical symbol for gold is Au.
```
- mlx (0.12s):
```
The chemical symbol for gold is Au.
```

### en-cs-04 — src ✓ / mlx ✓ / agree NO
- gold: Shakespeare
- source (0.66s):
```
The play "Romeo and Juliet" was written by the renowned English playwright William Shakespeare. It is one of his most famous and iconic works, and is widely considered a classic of world literature.
```
- mlx (0.25s):
```
The play "Romeo and Juliet" was written by the famous English playwright William Shakespeare. It is one of his most famous and iconic works, and is considered a tragedy.
```

### en-cs-05 — src ✓ / mlx ✓ / agree NO
- gold: Jupiter
- source (0.7s):
```
The largest planet in our solar system is Jupiter. It is a gas giant, meaning it is primarily composed of hydrogen and helium gases. Jupiter is the fifth planet from the Sun and is the largest of all the planets in terms of both mass and size.
```
- mlx (0.39s):
```
The largest planet in our solar system is Jupiter. It has a diameter of approximately 142,984 kilometers (88,846 miles). Jupiter is a gas giant, meaning it is primarily composed of hydrogen and helium gases. It is the fifth planet from the Sun and is known for its distinctive banded appearance, which is caused by strong winds in its atmosphere.
```

### en-cs-06 — src ✗ / mlx ✗ / agree NO
- gold: 0
- source (0.56s):
```
Water freezes at 0 degrees Celsius (°C) or 32 degrees Fahrenheit (°F).
```
- mlx (0.14s):
```
Water freezes at 0 degrees Celsius (or 32 degrees Fahrenheit).
```

### en-cs-07 — src ✗ / mlx ✓ / agree NO
- gold: 7
- source (0.63s):
```
There are 7 continents on Earth:

1. Africa
2. Antarctica
3. Asia
4. Australia
5. Europe
6. North America
7. South America
```
- mlx (0.24s):
```
There are seven continents on Earth:

1. Africa
2. Antarctica
3. Asia
4. Australia
5. Europe
6. North America
7. South America
```

### en-cs-08 — src ✓ / mlx ✓ / agree NO
- gold: George Washington
- source (0.6s):
```
George Washington was the first President of the United States. He served from April 30, 1789, to March 4, 1797.
```
- mlx (0.28s):
```
The first President of the United States was George Washington. He was inaugurated as the first President on April 30, 1789, and served two terms in office until March 4, 1797.
```

### en-cs-09 — src ✓ / mlx ✓ / agree NO
- gold: cold
- source (0.52s):
```
The opposite of the word "hot" is typically "cold".
```
- mlx (0.29s):
```
The opposite of the word "hot" is actually "cold". This is because "hot" and "cold" are two extremes on the temperature scale, with "hot" being the warmer and "cold" being the cooler.
```

### en-cs-10 — src ✓ / mlx ✓ / agree yes
- gold: 8
- source (0.51s):
```
Spiders have eight legs.
```
- mlx (0.11s):
```
Spiders have eight legs.
```

### math-01 — src ✗ / mlx ✗ / agree NO
- gold: 43
- source (0.61s):
```
To calculate 17 + 26, we need to add the two numbers together.

17 + 26 = 43

The result is 43.
```
- mlx (0.23s):
```
To calculate 17 + 26, we need to add the two numbers together.

17 + 26 = 43

The result of the addition is 43.
```

### math-02 — src ✗ / mlx ✗ / agree NO
- gold: 144
- source (0.62s):
```
To calculate 12 × 12, we multiply 12 by 12.

12 × 12 = 144

The result is 144.
```
- mlx (0.24s):
```
To calculate 12 × 12, we multiply 12 by 12.

12 × 12 = 144

So, 12 × 12 equals 144.
```

### math-03 — src ✗ / mlx ✗ / agree NO
- gold: 12
- source (0.63s):
```
To calculate 144 ÷ 12, we can simply divide 144 by 12.

144 ÷ 12 = 12

So the result is 12.
```
- mlx (0.24s):
```
To calculate 144 ÷ 12, we need to perform the division operation.

144 ÷ 12 = 12

So, the result of the division is 12.
```

### math-04 — src ✗ / mlx ✗ / agree NO
- gold: 30
- source (0.66s):
```
To find 15% of 200, you can multiply 200 by 0.15.

200 * 0.15 = 30

So, 15% of 200 is 30.
```
- mlx (0.26s):
```
To find 15% of 200, you multiply 200 by 0.15.

200 * 0.15 = 30

So, 15% of 200 is 30.
```

### math-05 — src ✗ / mlx ✗ / agree yes
- gold: 9
- source (0.52s):
```
The square root of 81 is 9.
```
- mlx (0.14s):
```
The square root of 81 is 9.
```

### math-06 — src ✗ / mlx ✗ / agree NO
- gold: 1024
- source (0.98s):
```
2 raised to the power of 10 can be calculated as follows:

2^10 = 2 multiplied by itself 10 times:

2 × 2 = 4
4 × 2 = 8
8 × 2 = 16
16 × 2 = 32
32 × 2 = 64
64 × 2 = 128
128 × 2 = 256
256 × 2 = 512
512 × 2 = 1024

So, 2 raised to the power of 10 is 1024.
```
- mlx (1.03s):
```
2 raised to the power of 10 is 2^10.

To calculate this, we multiply 2 by itself 10 times:

2 * 2 = 4
4 * 2 = 8
8 * 2 = 16
16 * 2 = 32
32 * 2 = 64
64 * 2 = 128
128 * 2 = 256
256 * 2 = 512
512 * 2 = 1024
1024 * 2 = 2048
2048 * 2 = 4096
4096 * 2 = 8192
8192 * 2 = 16384
16384 * 2 = 32768
32768 * 2 = 65536
65536 * 2 = 131072
131072 * 2 = 262144
262144 * 2 = 524288
524288 * 2 = 1048576

So, 2 raised to the power of 10 is 1048576.
```

### math-07 — src ✗ / mlx ✗ / agree NO
- gold: x = 4
- source (0.84s):
```
To solve for x, we need to isolate x on one side of the equation.

First, subtract 9 from both sides:

3x + 9 - 9 = 21 - 9
3x = 12

Next, divide both sides by 3:

3x / 3 = 12 / 3
x = 4

So, the value of x is 4.
```
- mlx (0.45s):
```
To solve for x, we need to isolate x on one side of the equation.

First, let's subtract 9 from both sides:

3x + 9 - 9 = 21 - 9
3x = 12

Next, let's divide both sides by 3:

3x / 3 = 12 / 3
x = 4

So, the value of x is 4.
```

### math-08 — src ✗ / mlx ⚠truncated / agree NO
- gold: 12
- source (1.34s):
```
To find the least common multiple (LCM) of 4 and 6, we need to find the smallest number that both 4 and 6 can divide into evenly.

The multiples of 4 are: 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, ...

The multiples of 6 are: 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, ...

The first number that appears in both lists is 12, so the least common multiple of 4 and 6 is 12.
```
- mlx (2.32s):
```
To find the least common multiple (LCM) of 4 and 6, we need to find the smallest number that both 4 and 6 can divide into evenly.

The multiples of 4 are: 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 464, 468, 472, 476, 480, 484, 488, 492, 496, 500, 504, 508, 512, 516, 520, 524, 528, 532, 536, 540, 544, 548, 552, 556, 560, 564, 568, 572, 576, 580, 584, 588, 592, 596, 600, 604, 608, 612, 616, 620, 624,
```

### math-09 — src ✗ / mlx ✗ / agree NO
- gold: 125
- source (0.65s):
```
To calculate 1000 ÷ 8, we can use division.

1000 ÷ 8 = 125

So the result of 1000 divided by 8 is 125.
```
- mlx (0.26s):
```
To calculate 1000 ÷ 8, we need to perform division.

1000 ÷ 8 = 125

So, 1000 divided by 8 equals 125.
```

### math-10 — src ✓ / mlx ✓ / agree yes
- gold: yes
- source (0.48s):
```
Yes
```
- mlx (0.11s):
```
Yes.
```

### logic-01 — src ✗ / mlx ✓ / agree NO
- gold: yes
- source (0.46s):
```
No.
```
- mlx (0.11s):
```
Yes.
```

### logic-02 — src ✓ / mlx ✓ / agree NO
- gold: 小刚
- source (0.81s):
```
根据问题的描述，我们可以推断出：

* 小明比小红高
* 小红比小刚高

这意味着小红比小刚矮。由于小红比小刚矮，我们可以推断出小红矮于小刚。

因此，答案是：小红矮于小刚。
```
- mlx (0.48s):
```
这三个人的身高顺序是：小红 > 小明 > 小刚。

小红比小明高，因为小红的身高比小明小。
小明比小红高，这是因为小明的身高比小红小。
小红比小刚高，这是因为小红的身高比小刚小。

所以，三人中谁最矮？小红最矮。
```

### logic-03 — src ✗ / mlx ✗ / agree NO
- gold: Saturday
- source (0.68s):
```
If today is Wednesday, 10 days from now will be:

Wednesday (today) + 10 days = Wednesday + 1 week = Thursday

So, 10 days from now, it will be Thursday.
```
- mlx (0.15s):
```
If today is Wednesday, 10 days from now will be a Friday.
```

### logic-04 — src ✓ / mlx ✓ / agree NO
- gold: C
- source (0.8s):
```
Let's analyze the information:

1. A is older than B.
2. B is older than C.

From point 1, we know A > B.
From point 2, we know B > C.

Now, let's combine the information:

A > B
B > C

This means A > C.

So, the youngest is C.
```
- mlx (0.09s):
```
C
```

### logic-05 — src ⚠truncated / mlx ✗ / agree NO
- gold: 6
- source (2.81s):
```
让我们一步一步地解决这个问题。

问题是：一个整数大于 5 且小于 7。

我们知道，整数的范围是 5 以下和 7 以上。

为了找到这个整数，我们可以从 6 开始，直到 6 以下。

6、7、8、9、10、11、12、13、14、15、16、17、18、19、20、21、22、23、24、25、26、27、28、29、30、31、32、33、34、35、36、37、38、39、40、41、42、43、44、45、46、47、48、49、50、51、52、53、54、55、56、57、58、59、60、61、62、63、64、65、66、67、68、69、70、71、72、73、74、75、76、77、78、79、80、81、82、83、84、85、86、87、88、89、90、91、92、93、94、95、96、97、98、99、100、101、102、103、104、105、106、107、108、109、110、111、112、113、114、115、116、117、118、119、120、121、122、123、124、125、126、127、128、129、130、131、132、133、134、135、136、137、138、139、140、141、142、143、144、145、146、147、148、149、150、151、152、153、154、155、156、157、158、159、160、161、162、163、164、165、166、167、168、169、170、171、172、173、174、175、176、177、178、179、180、181、182、183、184、185、186、187、188、189、190、191、192、193、194、195、196、197、198、199、200、201、202、203、204、205、206、207、208、209、210、211、212、213、214、215、216、217、218、219、220、221、222、223、224、225、226、227、228、229、230
```
- mlx (0.28s):
```
让我们一步一步地解决这个问题。

我们知道：

* 大于 5 的整数是 6
* 小于 7 的整数是 6

因此，我们的答案是 6。
```

### inst-01 — src ✓ / mlx ✓ / agree yes
- gold: blue (single word)
- source (0.49s):
```
Blue
```
- mlx (0.1s):
```
Blue
```

### inst-02 — src ✗ / mlx ✗ / agree yes
- gold: YES (bare)
- source (0.49s):
```
NO
```
- mlx (0.09s):
```
NO.
```

### inst-03 — src ✗ / mlx ✗ / agree NO
- gold: red, green, blue (all three present)
- source (0.51s):
```
Red, Blue, Yellow
```
- mlx (0.13s):
```
Red, Blue, and Yellow
```

### inst-04 — src ✗ / mlx ⚠repetition_loop / agree NO
- gold: 熊猫（≤5字）
- source (0.54s):
```
象、狮、豹、狒、鹦
```
- mlx (0.77s):
```
中国的国宝动物是象、龙、象牙、狮、狐、狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、狒狒、
```

### inst-05 — src ✗ / mlx ✗ / agree NO
- gold: 48
- source (0.49s):
```
24
```
- mlx (0.09s):
```
14
```

### open-01 — src ✓ / mlx ✓ / agree NO
- gold: (open — anomaly screening only)
- source (0.64s):
```
Autumn brings a kaleidoscope of colors to the landscape, as the vibrant hues of orange, red, and yellow paint the sky and trees with a warm, golden light.
```
- mlx (0.21s):
```
Autumn is a season of vibrant colors, crisp air, and cozy sweaters, as the leaves turn golden and crimson before falling to the ground.
```

### open-02 — src ✓ / mlx ✓ / agree NO
- gold: (open — anomaly screening only)
- source (0.7s):
```
春天是一年中最美丽的时期，充满了阳光、花 blossoms 和温暖的气息。它是人生最快乐的时期，充满了新 beginnings 和成长的机会。
```
- mlx (0.25s):
```
春天是人類最 favourite 的季節，天氣溫暖，天花花綠綠，花朵開放，人們喜歡到這個時候。
```

### open-03 — src ✓ / mlx ✓ / agree NO
- gold: (open — anomaly screening only)
- source (0.73s):
```
People wear coats in winter to protect themselves from the cold temperatures, which can range from -20°C to -50°C (−4°F to −58°F) in extreme cases. The coat serves as a barrier between the wearer's skin and the cold air, keeping them warm and comfortable.
```
- mlx (0.26s):
```
People wear coats in winter to protect themselves from the cold temperatures and to provide insulation and warmth. The coat also serves as a barrier between the wearer and the cold, keeping them dry and comfortable.
```

### open-04 — src ✓ / mlx ⚠repetition_loop,truncated / agree NO
- gold: (open — anomaly screening only)
- source (0.81s):
```
我喜欢的书是《《三国演义》》。这本书是中国历史上最著名的传记书之一，讲述了三国时期的历史，尤其是蜀汉时期的历史。它的内容丰富，写作流畅，具有深远的影响力。
```
- mlx (2.34s):
```
我喜欢的书是“人生不分情境”的《人生不分情境》。这本书由马克思、列尔、克里斯托夫和马克思的儿子马克思的儿子、马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的儿子马克思的
```

### open-05 — src ✓ / mlx ✓ / agree NO
- gold: (open — anomaly screening only)
- source (0.57s):
```
I don't have a favorite color, but I can tell you about some popular colors if you'd like.
```
- mlx (0.22s):
```
My favorite color is blue, as it's often associated with trust, loyalty, and calmness, which are qualities I strive to embody in my interactions with users.
```

