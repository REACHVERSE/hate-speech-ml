import math
def sigmoid(x):
    if x < 0.0:
        z = math.exp(x)
        return z / (1.0 + z)
    return 1.0 / (1.0 + math.exp(-x))
def score(input):
    if input[7224] < 0.29038507:
        if input[6434] >= 0.29639363:
            var0 = -0.01818182
        else:
            if input[5869] >= 0.22702667:
                var0 = -0.05
            else:
                if input[8239] >= 0.16533785:
                    if input[2658] < 0.31858608:
                        var0 = 0.1
                    else:
                        if input[7224] < 0.14709711:
                            var0 = -0.05
                        else:
                            var0 = 0.040000003
                else:
                    if input[1854] >= 0.1908037:
                        if input[6785] < 0.114068605:
                            var0 = 0.11111112
                        else:
                            var0 = -0.06666667
                    else:
                        if input[8626] >= 0.15316723:
                            var0 = 0.06666667
                        else:
                            var0 = 0.18867403
    else:
        if input[5997] >= 0.13479984:
            if input[5280] < 0.2012814:
                if input[6042] < 0.28997013:
                    var0 = 0.0
                else:
                    var0 = -0.14400001
            else:
                if input[8239] < 0.27070242:
                    if input[5997] < 0.13610455:
                        var0 = 0.0
                    else:
                        if input[5997] < 0.14727856:
                            var0 = -0.07368421
                        else:
                            var0 = -0.15789475
                else:
                    if input[6785] >= 0.12355902:
                        if input[6896] < 1.002881:
                            var0 = 0.0
                        else:
                            var0 = -0.12307693
                    else:
                        if input[5997] < 0.20598951:
                            var0 = 0.17669903
                        else:
                            var0 = -0.05185185
        else:
            if input[3844] < 1.1679463:
                if input[6785] >= 0.0974956:
                    if input[3825] < 0.27538994:
                        if input[3825] < 0.18823664:
                            var0 = -0.040000003
                        else:
                            var0 = -0.16000001
                    else:
                        if input[3844] < 0.23199886:
                            var0 = 0.05631068
                        else:
                            var0 = -0.061538465
                else:
                    if input[3844] < 0.13576418:
                        if input[450] >= 0.12612756:
                            var0 = 0.0
                        else:
                            var0 = 0.19237058
                    else:
                        if input[9307] >= 0.1884855:
                            var0 = -0.030769233
                        else:
                            var0 = 0.12380953
            else:
                if input[5496] < 0.7983319:
                    if input[9526] < 0.4657235:
                        var0 = -0.1
                    else:
                        if input[8239] >= 0.10807894:
                            var0 = 0.095
                        else:
                            var0 = 0.16855577
                else:
                    if input[6785] < 0.073931284:
                        if input[886] < 0.48912883:
                            var0 = -0.1
                        else:
                            var0 = 0.18698482
                    else:
                        if input[710] >= 0.22767049:
                            var0 = 0.19587629
                        else:
                            var0 = -0.06306439
    if input[7224] < 0.38224143:
        if input[6785] < 0.1552051:
            if input[3898] >= 0.22453628:
                var1 = 0.011829457
            else:
                if input[8239] >= 0.18935817:
                    var1 = 0.01968255
                else:
                    if input[809] >= 0.19132125:
                        var1 = 0.056493428
                    else:
                        if input[1020] < 0.34680095:
                            var1 = 0.056493428
                        else:
                            var1 = 0.18027191
        else:
            if input[7277] >= 0.26378962:
                var1 = -0.07465672
            else:
                if input[2859] >= 0.24136442:
                    var1 = -0.073702775
                else:
                    if input[1854] >= 0.1908037:
                        var1 = -0.0630032
                    else:
                        if input[9916] >= 0.13446148:
                            var1 = 0.004195498
                        else:
                            var1 = 0.15270266
    else:
        if input[5997] >= 0.13479984:
            if input[5280] < 0.2012814:
                if input[6042] < 0.28997013:
                    var1 = 0.0
                else:
                    var1 = -0.13139938
            else:
                if input[8239] < 0.27070242:
                    if input[5997] < 0.13610455:
                        var1 = 0.0
                    else:
                        if input[8239] < 0.13896155:
                            var1 = -0.067092426
                        else:
                            var1 = -0.14698373
                else:
                    if input[6785] >= 0.12355902:
                        if input[6896] < 1.002881:
                            var1 = 0.0
                        else:
                            var1 = -0.11303713
                    else:
                        if input[5997] < 0.20598951:
                            var1 = 0.16037814
                        else:
                            var1 = -0.047462992
        else:
            if input[3844] < 1.1679463:
                if input[6785] >= 0.0974956:
                    if input[3825] < 0.27538994:
                        if input[3825] < 0.18823664:
                            var1 = -0.037609342
                        else:
                            var1 = -0.14798169
                    else:
                        if input[3844] < 0.23199886:
                            var1 = 0.050938535
                        else:
                            var1 = -0.056378137
                else:
                    if input[3844] < 0.13848132:
                        if input[450] >= 0.12612756:
                            var1 = 0.0
                        else:
                            var1 = 0.17348573
                    else:
                        if input[9307] >= 0.1884855:
                            var1 = -0.028171513
                        else:
                            var1 = 0.11102287
            else:
                if input[5496] < 0.7983319:
                    if input[1854] >= 0.20459293:
                        var1 = -0.019774629
                    else:
                        if input[6785] >= 0.11275789:
                            var1 = 0.023066496
                        else:
                            var1 = 0.14919104
                else:
                    if input[5280] < 0.06627581:
                        if input[2111] >= 0.0937234:
                            var1 = -0.047664095
                        else:
                            var1 = 0.18159519
                    else:
                        if input[710] >= 0.22767049:
                            var1 = 0.17823355
                        else:
                            var1 = -0.056719836
    if input[7224] < 0.29038507:
        if input[6785] < 0.1552051:
            if input[3898] >= 0.22453628:
                var2 = 0.011182322
            else:
                if input[8239] >= 0.18935817:
                    var2 = 0.018595647
                else:
                    if input[5113] >= 0.29979643:
                        var2 = 0.047272883
                    else:
                        if input[6053] < 0.35879207:
                            var2 = 0.047272883
                        else:
                            var2 = 0.16698019
        else:
            if input[7277] >= 0.26378962:
                var2 = -0.07036189
            else:
                if input[6434] >= 0.29639363:
                    var2 = -0.070090495
                else:
                    if input[2201] < 0.2552148:
                        var2 = -0.05156691
                    else:
                        if input[8576] >= 0.18029012:
                            var2 = -0.048314303
                        else:
                            var2 = 0.14228673
    else:
        if input[5280] >= 0.07952901:
            if input[4170] < 1.705911:
                if input[316] < 1.1247389:
                    if input[9213] < 0.19799241:
                        if input[316] < 0.17714117:
                            var2 = 0.124957696
                        else:
                            var2 = 0.028926466
                    else:
                        if input[1783] < 0.3843465:
                            var2 = 0.07345944
                        else:
                            var2 = -0.1389597
                else:
                    if input[360] < 0.92302686:
                        var2 = -0.1441867
                    else:
                        if input[284] >= 0.16601215:
                            var2 = -0.11993231
                        else:
                            var2 = 0.06482007
            else:
                if input[3000] < 0.357766:
                    if input[3480] >= 0.19949864:
                        var2 = -0.13825579
                    else:
                        if input[8453] < 0.2830065:
                            var2 = 0.1469882
                        else:
                            var2 = 0.02380239
                else:
                    if input[962] < 0.51143813:
                        var2 = 0.15394185
                    else:
                        if input[963] < 0.7767438:
                            var2 = 0.115683116
                        else:
                            var2 = -0.12727346
        else:
            if input[4607] >= 0.15140015:
                if input[4607] < 0.2510862:
                    if input[6785] < 0.16000375:
                        var2 = 0.023872932
                    else:
                        if input[2957] >= 0.18397737:
                            var2 = -0.06013949
                        else:
                            var2 = -0.18761669
                else:
                    if input[4607] < 0.32510442:
                        var2 = -0.018256446
                    else:
                        var2 = 0.07068436
            else:
                if input[5997] >= 0.13479984:
                    if input[8239] < 0.27070242:
                        if input[5997] < 0.13610455:
                            var2 = 0.0
                        else:
                            var2 = -0.11120363
                    else:
                        if input[6785] >= 0.12355902:
                            var2 = -0.09034653
                        else:
                            var2 = 0.14291404
                else:
                    if input[8239] < 0.082418025:
                        if input[3609] >= 0.073660225:
                            var2 = 0.032820724
                        else:
                            var2 = 0.18939057
                    else:
                        if input[6785] < 0.073931284:
                            var2 = 0.16805726
                        else:
                            var2 = -0.013477147
    if input[7224] < 0.38224143:
        if input[6785] < 0.1552051:
            if input[3898] >= 0.22453628:
                var3 = 0.01057092
            else:
                if input[8239] >= 0.18935817:
                    var3 = 0.017570524
                else:
                    if input[5655] >= 0.1966982:
                        var3 = 0.040256783
                    else:
                        if input[9379] >= 0.14184728:
                            var3 = 0.040256783
                        else:
                            var3 = 0.15603088
        else:
            if input[7224] < 0.17183623:
                if input[2201] >= 0.2229781:
                    var3 = -0.052524615
                else:
                    if input[3916] >= 0.10828049:
                        if input[3609] < 0.3962766:
                            var3 = 0.10623144
                        else:
                            var3 = -0.06864572
                    else:
                        if input[284] >= 0.16551289:
                            var3 = -0.0042696176
                        else:
                            var3 = 0.14721946
            else:
                if input[3641] < 0.31751505:
                    if input[8626] < 0.20916337:
                        var3 = -0.032768134
                    else:
                        if input[3641] < 0.14984073:
                            var3 = 0.041609198
                        else:
                            var3 = 0.13642858
                else:
                    if input[7224] < 0.17684487:
                        var3 = -0.056306608
                    else:
                        if input[9679] < 0.27542502:
                            var3 = 0.11166499
                        else:
                            var3 = 0.03334794
    else:
        if input[5280] >= 0.086441964:
            if input[4170] >= 0.14530462:
                if input[316] >= 0.1711332:
                    if input[9213] < 0.19799241:
                        var3 = 0.07850279
                    else:
                        if input[7107] >= 0.26211184:
                            var3 = 0.05599469
                        else:
                            var3 = -0.13328587
                else:
                    if input[360] < 0.92302686:
                        var3 = -0.117181994
                    else:
                        if input[284] < 0.2431559:
                            var3 = -0.11311495
                        else:
                            var3 = 0.056550663
            else:
                if input[3000] < 0.357766:
                    if input[3480] < 0.34458226:
                        var3 = -0.12600909
                    else:
                        if input[3047] < 0.28047752:
                            var3 = -0.09476062
                        else:
                            var3 = 0.06155612
                else:
                    if input[3371] < 0.25352657:
                        if input[3371] < 0.1997008:
                            var3 = 0.13238072
                        else:
                            var3 = 0.054530647
                    else:
                        if input[962] < 0.51143813:
                            var3 = 0.12812237
                        else:
                            var3 = -0.1172436
        else:
            if input[4607] >= 0.15140015:
                if input[4607] < 0.2510862:
                    if input[6785] < 0.16000375:
                        var3 = 0.022545887
                    else:
                        if input[4607] < 0.21041769:
                            var3 = -0.17251748
                        else:
                            var3 = -0.055514093
                else:
                    if input[4607] < 0.32510442:
                        var3 = -0.017251259
                    else:
                        var3 = 0.06671536
            else:
                if input[5997] >= 0.13479984:
                    if input[8239] < 0.27070242:
                        if input[8239] < 0.13896155:
                            var3 = -0.04430611
                        else:
                            var3 = -0.13049498
                    else:
                        if input[6785] >= 0.12355902:
                            var3 = -0.08335564
                        else:
                            var3 = 0.13251428
                else:
                    if input[5496] < 0.4169824:
                        if input[9526] < 0.4657235:
                            var3 = -0.10983645
                        else:
                            var3 = 0.13569777
                    else:
                        if input[6785] < 0.073931284:
                            var3 = 0.1569443
                        else:
                            var3 = -0.015683
    if input[7224] < 0.29038507:
        if input[6785] < 0.1552051:
            if input[3898] >= 0.22453628:
                var4 = 0.009993202
            else:
                if input[8239] >= 0.18935817:
                    var4 = 0.016603371
                else:
                    if input[749] >= 0.18179011:
                        var4 = 0.032813367
                    else:
                        if input[2957] >= 0.18597658:
                            var4 = 0.032813367
                        else:
                            var4 = 0.14781943
        else:
            if input[2859] >= 0.24136442:
                var4 = -0.07895298
            else:
                if input[7277] >= 0.26378962:
                    var4 = -0.068336494
                else:
                    if input[1854] >= 0.1908037:
                        var4 = -0.063678496
                    else:
                        if input[9679] >= 0.27080962:
                            var4 = -0.05593444
                        else:
                            var4 = 0.12367888
    else:
        if input[5280] >= 0.07952901:
            if input[4170] < 1.705911:
                if input[316] < 1.1247389:
                    if input[9213] < 0.19799241:
                        if input[316] < 0.17714117:
                            var4 = 0.113341294
                        else:
                            var4 = 0.022909088
                    else:
                        if input[1783] < 0.3843465:
                            var4 = 0.07677187
                        else:
                            var4 = -0.11855667
                else:
                    if input[360] < 0.92302686:
                        var4 = -0.12854417
                    else:
                        if input[2580] < 0.7530025:
                            var4 = -0.117032945
                        else:
                            var4 = 0.053734828
            else:
                if input[3000] < 0.2644257:
                    if input[3480] >= 0.19949864:
                        var4 = -0.1194453
                    else:
                        if input[8453] < 0.23216556:
                            var4 = 0.14660683
                        else:
                            var4 = 0.03343102
                else:
                    if input[962] < 0.51143813:
                        var4 = 0.13322096
                    else:
                        if input[963] < 0.7767438:
                            var4 = 0.114160635
                        else:
                            var4 = -0.10651908
        else:
            if input[4607] >= 0.15140015:
                if input[4607] < 0.2510862:
                    if input[6785] < 0.16000375:
                        var4 = 0.021295872
                    else:
                        if input[2957] >= 0.18397737:
                            var4 = -0.049300607
                        else:
                            var4 = -0.16093795
                else:
                    if input[4607] < 0.32510442:
                        var4 = -0.016302774
                    else:
                        var4 = 0.063053526
            else:
                if input[2020] >= 0.17728917:
                    if input[2020] < 0.22602095:
                        if input[8239] < 0.23995198:
                            var4 = -0.058587104
                        else:
                            var4 = -0.18273242
                    else:
                        if input[2020] < 0.27830365:
                            var4 = 0.09577873
                        else:
                            var4 = -0.045243762
                else:
                    if input[3609] >= 0.104537666:
                        if input[6785] < 0.2709589:
                            var4 = 0.031591937
                        else:
                            var4 = -0.14980257
                    else:
                        if input[9916] >= 0.1284083:
                            var4 = -0.10799891
                        else:
                            var4 = 0.02590029
    if input[7224] < 0.38224143:
        if input[6785] < 0.22547413:
            if input[8385] >= 0.18800016:
                var5 = -0.011059648
            else:
                if input[3898] >= 0.23400651:
                    var5 = -0.0013345006
                else:
                    if input[8626] >= 0.15756905:
                        if input[3609] >= 0.18332712:
                            var5 = -0.018651785
                        else:
                            var5 = 0.0775311
                    else:
                        if input[9379] >= 0.10025676:
                            var5 = 0.06423546
                        else:
                            var5 = 0.14201953
        else:
            if input[7224] < 0.17183623:
                if input[2201] < 0.2552148:
                    var5 = -0.05354005
                else:
                    if input[3916] >= 0.10828049:
                        if input[3609] < 0.3962766:
                            var5 = 0.09561249
                        else:
                            var5 = -0.071323864
                    else:
                        if input[9970] < 0.27321562:
                            var5 = -0.019201085
                        else:
                            var5 = 0.13140269
            else:
                if input[3609] < 0.20596567:
                    var5 = 0.109465234
                else:
                    if input[9468] < 0.3867093:
                        var5 = 0.0896554
                    else:
                        if input[9679] < 0.19938773:
                            var5 = 0.096581176
                        else:
                            var5 = -0.04795845
    else:
        if input[5997] >= 0.13479984:
            if input[5280] < 0.2012814:
                if input[9379] >= 0.18174723:
                    var5 = -0.0072871777
                else:
                    var5 = -0.10151587
            else:
                if input[8239] < 0.27070242:
                    if input[5997] < 0.14727856:
                        if input[8239] < 0.13508217:
                            var5 = -0.09040636
                        else:
                            var5 = 0.060336094
                    else:
                        var5 = -0.12476388
                else:
                    if input[9679] >= 0.114979394:
                        if input[5997] < 0.15704294:
                            var5 = -0.0019107986
                        else:
                            var5 = -0.13265626
                    else:
                        if input[6785] >= 0.12355902:
                            var5 = -0.068839245
                        else:
                            var5 = 0.12780318
        else:
            if input[3844] < 0.41781428:
                if input[6785] >= 0.10946473:
                    if input[3825] < 0.27538994:
                        var5 = -0.13920288
                    else:
                        if input[3875] < 1.0078187:
                            var5 = -0.1120768
                        else:
                            var5 = 0.02753953
                else:
                    if input[3844] < 0.14459579:
                        if input[9379] >= 0.10629529:
                            var5 = -0.052497424
                        else:
                            var5 = 0.15609449
                    else:
                        if input[3825] < 0.182618:
                            var5 = -0.08614575
                        else:
                            var5 = 0.1014895
            else:
                if input[710] >= 0.22767049:
                    var5 = 0.16491662
                else:
                    if input[5736] < 0.98390526:
                        var5 = 0.19373767
                    else:
                        if input[5280] < 0.06627581:
                            var5 = 0.1553369
                        else:
                            var5 = -0.042097826
    if input[7224] < 0.38224143:
        if input[6785] < 0.22547413:
            if input[8385] >= 0.18800016:
                var6 = -0.010456461
            else:
                if input[3898] >= 0.23400651:
                    var6 = -0.0012621641
                else:
                    if input[8626] >= 0.15756905:
                        if input[3609] >= 0.18332712:
                            var6 = -0.01752902
                        else:
                            var6 = 0.07349554
                    else:
                        if input[9379] >= 0.10025676:
                            var6 = 0.06022187
                        else:
                            var6 = 0.13609433
        else:
            if input[7224] < 0.17183623:
                if input[2201] >= 0.2229781:
                    var6 = -0.054121096
                else:
                    if input[3916] >= 0.10828049:
                        if input[3609] < 0.3962766:
                            var6 = 0.09157303
                        else:
                            var6 = -0.06594705
                    else:
                        if input[8626] >= 0.10695634:
                            var6 = 0.004524078
                        else:
                            var6 = 0.12583007
            else:
                if input[3609] < 0.20596567:
                    if input[7224] < 0.17929856:
                        if input[3641] < 0.20311493:
                            var6 = 0.082540244
                        else:
                            var6 = -0.011003369
                    else:
                        var6 = 0.11500589
                else:
                    if input[9679] < 0.27542502:
                        var6 = 0.09713319
                    else:
                        if input[9468] < 0.3867093:
                            var6 = 0.08025555
                        else:
                            var6 = -0.04310253
    else:
        if input[9379] < 0.17845324:
            if input[6321] < 0.23743644:
                var6 = 0.2141234
            else:
                if input[5280] >= 0.05972259:
                    if input[8239] < 0.114009485:
                        var6 = 0.07116083
                    else:
                        if input[8453] >= 0.1737074:
                            var6 = 0.060690887
                        else:
                            var6 = -0.1005803
                else:
                    if input[9379] < 0.10629529:
                        var6 = 0.1813494
                    else:
                        if input[2762] < 0.22394764:
                            var6 = 0.14223981
                        else:
                            var6 = 0.024831166
        else:
            if input[9679] < 0.09171186:
                if input[3916] >= 0.07751899:
                    if input[3916] < 0.09497498:
                        var6 = -0.075896606
                    else:
                        var6 = 0.034985572
                else:
                    if input[316] >= 0.12744412:
                        var6 = -0.110564195
                    else:
                        if input[6963] >= 0.13917075:
                            var6 = 0.0027201695
                        else:
                            var6 = 0.18174352
            else:
                if input[3844] < 1.1679463:
                    if input[6785] >= 0.10946473:
                        if input[3825] < 0.27538994:
                            var6 = -0.13080288
                        else:
                            var6 = 0.01556695
                    else:
                        if input[3844] < 0.14459579:
                            var6 = 0.147368
                        else:
                            var6 = 0.08511042
                else:
                    if input[710] >= 0.22767049:
                        var6 = 0.1544849
                    else:
                        if input[5997] >= 0.13479984:
                            var6 = 0.08967287
                        else:
                            var6 = -0.039893996
    if input[7224] < 0.29038507:
        if input[6785] < 0.1552051:
            if input[3898] >= 0.22453628:
                var7 = 0.0035640288
            else:
                if input[8239] >= 0.18935817:
                    var7 = 0.005539158
                else:
                    if input[749] >= 0.18179011:
                        var7 = 0.0196453
                    else:
                        if input[6053] < 0.35879207:
                            var7 = 0.020066509
                        else:
                            var7 = 0.13016717
        else:
            if input[7224] < 0.12369221:
                if input[8466] >= 0.10038626:
                    var7 = -0.016613966
                else:
                    var7 = 0.13305199
            else:
                if input[3641] >= 0.1348824:
                    if input[2201] < 0.27155608:
                        var7 = -0.07182052
                    else:
                        if input[8626] >= 0.12920874:
                            var7 = -0.036726743
                        else:
                            var7 = 0.12514864
                else:
                    if input[1126] < 0.28528884:
                        if input[7224] < 0.12779771:
                            var7 = 0.021675255
                        else:
                            var7 = 0.118474066
                    else:
                        if input[9468] < 0.19219531:
                            var7 = 0.101325646
                        else:
                            var7 = -0.015600701
    else:
        if input[5280] >= 0.08755518:
            if input[4170] < 1.705911:
                if input[316] >= 0.16760005:
                    if input[9213] < 0.19799241:
                        if input[316] < 0.17714117:
                            var7 = 0.11160469
                        else:
                            var7 = 0.026207859
                    else:
                        if input[1783] < 0.3843465:
                            var7 = 0.077080674
                        else:
                            var7 = -0.1058794
                else:
                    if input[284] < 0.2431559:
                        var7 = -0.11300629
                    else:
                        if input[360] < 0.92302686:
                            var7 = -0.10433232
                        else:
                            var7 = 0.053249415
            else:
                if input[3000] < 0.357766:
                    if input[3480] < 0.34458226:
                        var7 = -0.10500225
                    else:
                        if input[9679] >= 0.12251225:
                            var7 = -0.07772105
                        else:
                            var7 = 0.06269495
                else:
                    if input[5372] < 1.2109509:
                        if input[5280] < 0.10766058:
                            var7 = 0.045755904
                        else:
                            var7 = 0.18707493
                    else:
                        if input[3371] < 0.34279203:
                            var7 = 0.08558356
                        else:
                            var7 = -0.09449227
        else:
            if input[4607] >= 0.15140015:
                if input[4607] < 0.2510862:
                    if input[6785] < 0.16000375:
                        var7 = 0.021947663
                    else:
                        if input[2957] >= 0.18397737:
                            var7 = -0.04277374
                        else:
                            var7 = -0.14763217
                else:
                    if input[4607] < 0.32510442:
                        var7 = -0.013135073
                    else:
                        var7 = 0.06408501
            else:
                if input[2020] >= 0.17728917:
                    if input[2020] < 0.22602095:
                        if input[8239] < 0.23995198:
                            var7 = -0.051282253
                        else:
                            var7 = -0.16278087
                    else:
                        if input[2020] < 0.27830365:
                            var7 = 0.094879225
                        else:
                            var7 = -0.037530925
                else:
                    if input[8626] >= 0.1311385:
                        if input[6785] >= 0.09048581:
                            var7 = -0.008374483
                        else:
                            var7 = -0.13540147
                    else:
                        if input[3609] >= 0.104537666:
                            var7 = -0.10888016
                        else:
                            var7 = 0.02341587
    if input[7224] < 0.38224143:
        if input[6785] < 0.22547413:
            if input[8385] >= 0.18800016:
                var8 = -0.012263818
            else:
                if input[8626] >= 0.15756905:
                    if input[8626] < 0.2032789:
                        if input[3609] >= 0.17309473:
                            var8 = -0.04545241
                        else:
                            var8 = 0.032663014
                    else:
                        var8 = 0.08324404
                else:
                    if input[3898] >= 0.23400651:
                        var8 = -0.0038060609
                    else:
                        if input[9379] >= 0.10025676:
                            var8 = 0.05020706
                        else:
                            var8 = 0.12716112
        else:
            if input[7224] < 0.17183623:
                if input[8626] >= 0.10695634:
                    if input[3609] < 0.135883:
                        var8 = 0.03622741
                    else:
                        var8 = -0.060470592
                else:
                    if input[3916] >= 0.10962784:
                        if input[9679] < 0.12505844:
                            var8 = 0.029021595
                        else:
                            var8 = -0.028289378
                    else:
                        if input[2201] < 0.2552148:
                            var8 = -0.03745414
                        else:
                            var8 = 0.11549252
            else:
                if input[3609] < 0.20596567:
                    if input[7224] < 0.17929856:
                        if input[3641] < 0.20311493:
                            var8 = 0.074760206
                        else:
                            var8 = -0.0136085
                    else:
                        var8 = 0.10572507
                else:
                    if input[9679] < 0.27542502:
                        var8 = 0.09204262
                    else:
                        if input[9468] < 0.3867093:
                            var8 = 0.07276718
                        else:
                            var8 = -0.039306402
    else:
        if input[5496] < 0.4169824:
            if input[9526] < 0.4657235:
                var8 = -0.10488664
            else:
                if input[6785] >= 0.10946473:
                    if input[5496] < 0.20886993:
                        if input[5496] < 0.19225287:
                            var8 = -0.009486205
                        else:
                            var8 = -0.037785627
                    else:
                        var8 = 0.05602243
                else:
                    if input[8239] >= 0.10807894:
                        if input[8424] < 0.19307473:
                            var8 = -0.13147853
                        else:
                            var8 = 0.08907077
                    else:
                        if input[9647] >= 0.2514336:
                            var8 = -0.047777418
                        else:
                            var8 = 0.13693282
        else:
            if input[6785] < 0.083166264:
                if input[6785] < 0.06657117:
                    if input[1897] < 0.15101352:
                        var8 = 0.021756055
                    else:
                        if input[1997] >= 0.06402852:
                            var8 = 0.037307654
                        else:
                            var8 = 0.15477325
                else:
                    if input[5280] < 0.0737354:
                        if input[6785] < 0.07553596:
                            var8 = -0.08622191
                        else:
                            var8 = -0.014528118
                    else:
                        if input[4411] >= 0.1734649:
                            var8 = -0.13963695
                        else:
                            var8 = 0.119771175
            else:
                if input[5997] >= 0.13479984:
                    if input[5280] < 0.2012814:
                        if input[9379] >= 0.18174723:
                            var8 = -0.009660776
                        else:
                            var8 = -0.09587529
                    else:
                        if input[8239] < 0.27070242:
                            var8 = -0.08837762
                        else:
                            var8 = 0.10257955
                else:
                    if input[9679] < 0.09221593:
                        if input[3916] >= 0.083009586:
                            var8 = -0.04627801
                        else:
                            var8 = 0.16611402
                    else:
                        if input[2344] < 0.2415315:
                            var8 = 0.18697314
                        else:
                            var8 = -0.033175122
    if input[7224] < 0.29038507:
        if input[6785] < 0.1552051:
            if input[8239] >= 0.18935817:
                var9 = 0.00077244453
            else:
                if input[3898] >= 0.22453628:
                    var9 = 0.0008527188
                else:
                    if input[8626] >= 0.15756905:
                        if input[8626] < 0.17404854:
                            var9 = -0.0127033265
                        else:
                            var9 = 0.07350206
                    else:
                        if input[3574] >= 0.15275511:
                            var9 = 0.0145621095
                        else:
                            var9 = 0.12289333
        else:
            if input[7224] < 0.12369221:
                if input[8466] >= 0.10038626:
                    var9 = -0.0224191
                else:
                    var9 = 0.1248458
            else:
                if input[3641] >= 0.1348824:
                    if input[8626] < 0.2032789:
                        var9 = -0.042760853
                    else:
                        if input[2201] < 0.35301754:
                            var9 = -0.04112497
                        else:
                            var9 = 0.11641019
                else:
                    if input[1126] < 0.28528884:
                        if input[9679] >= 0.10408659:
                            var9 = 0.022928553
                        else:
                            var9 = 0.109573446
                    else:
                        if input[9468] < 0.19219531:
                            var9 = 0.092743985
                        else:
                            var9 = -0.016877826
    else:
        if input[8239] < 0.082418025:
            if input[3609] >= 0.073660225:
                var9 = 0.03833327
            else:
                if input[8239] < 0.08123742:
                    if input[1997] >= 0.0895276:
                        var9 = 0.033716347
                    else:
                        var9 = 0.16827928
                else:
                    var9 = 0.061923344
        else:
            if input[5736] < 0.98390526:
                if input[8239] < 0.105920464:
                    var9 = 0.052630592
                else:
                    if input[2859] < 0.2703223:
                        var9 = 0.056585915
                    else:
                        if input[3269] < 0.28820807:
                            var9 = 0.05848477
                        else:
                            var9 = 0.18430041
            else:
                if input[263] >= 0.16165602:
                    if input[268] < 0.5073344:
                        if input[5997] < 0.14402059:
                            var9 = 0.124389306
                        else:
                            var9 = 0.21788071
                    else:
                        if input[264] < 0.5886371:
                            var9 = 0.1340792
                        else:
                            var9 = -0.031869844
                else:
                    if input[9379] < 0.17845324:
                        if input[6321] < 0.23743644:
                            var9 = 0.17409246
                        else:
                            var9 = 0.07066533
                    else:
                        if input[3844] < 0.2977522:
                            var9 = 0.09139592
                        else:
                            var9 = -0.03042234
    if input[6785] < 0.083166264:
        if input[4411] >= 0.1734649:
            var10 = -0.110589184
        else:
            if input[5280] >= 0.05972259:
                if input[5280] < 0.0737354:
                    if input[4170] < 0.11331123:
                        var10 = 0.06579724
                    else:
                        if input[6785] < 0.07637985:
                            var10 = -0.08568167
                        else:
                            var10 = -0.0153858615
                else:
                    var10 = 0.1380677
            else:
                if input[284] >= 0.1490245:
                    var10 = -0.08103367
                else:
                    if input[403] >= 0.07113375:
                        var10 = -0.052516054
                    else:
                        if input[8453] >= 0.13893692:
                            var10 = -0.05204698
                        else:
                            var10 = 0.13484184
    else:
        if input[5496] < 0.4169824:
            if input[9526] < 0.4657235:
                var10 = -0.099413425
            else:
                if input[6785] >= 0.10946473:
                    if input[5496] < 0.20886993:
                        if input[5496] < 0.18834554:
                            var10 = -0.034795567
                        else:
                            var10 = -0.011343226
                    else:
                        if input[5496] < 0.22405513:
                            var10 = 0.06451123
                        else:
                            var10 = 0.02082566
                else:
                    if input[4365] >= 0.22268921:
                        var10 = -0.057219423
                    else:
                        if input[9614] >= 0.24473223:
                            var10 = -0.054211237
                        else:
                            var10 = 0.12303406
        else:
            if input[5280] < 0.06235251:
                if input[6108] < 0.32275945:
                    var10 = -0.07760883
                else:
                    if input[9379] >= 0.07571638:
                        var10 = 0.044260398
                    else:
                        if input[4819] >= 0.028327148:
                            var10 = 0.04423214
                        else:
                            var10 = 0.16997421
            else:
                if input[7224] < 0.38224143:
                    if input[3641] < 0.23172016:
                        if input[2201] < 0.2552148:
                            var10 = -0.024497902
                        else:
                            var10 = 0.11782169
                    else:
                        if input[5018] < 0.31646013:
                            var10 = -0.08040611
                        else:
                            var10 = 0.081755385
                else:
                    if input[5997] >= 0.13479984:
                        if input[5997] < 0.20598951:
                            var10 = 0.088984944
                        else:
                            var10 = -0.10804472
                    else:
                        if input[710] >= 0.22767049:
                            var10 = 0.14835386
                        else:
                            var10 = -0.02986603
    if input[5280] >= 0.09344549:
        if input[4170] < 1.705911:
            if input[316] < 1.1247389:
                if input[7107] < 0.3501629:
                    var11 = 0.079166956
                else:
                    if input[654] < 0.3134469:
                        var11 = 0.06991183
                    else:
                        if input[9213] < 0.19799241:
                            var11 = 0.029994965
                        else:
                            var11 = -0.103641294
            else:
                if input[903] < 0.49725503:
                    if input[5280] < 0.10448626:
                        var11 = 0.020532986
                    else:
                        var11 = 0.14863257
                else:
                    if input[5050] < 1.1461158:
                        var11 = -0.10746342
                    else:
                        if input[4241] < 0.39811212:
                            var11 = 0.15281722
                        else:
                            var11 = 0.039954457
        else:
            if input[8453] < 0.2830065:
                if input[5997] < 0.17797771:
                    var11 = -0.06944141
                else:
                    if input[8466] >= 0.14605875:
                        var11 = -0.040937588
                    else:
                        if input[8458] < 0.33151075:
                            var11 = 0.051285084
                        else:
                            var11 = 0.13561003
            else:
                if input[963] >= 0.26904714:
                    if input[963] < 0.32194155:
                        var11 = 0.17522667
                    else:
                        if input[963] < 0.37806922:
                            var11 = -0.0151669085
                        else:
                            var11 = 0.0887655
                else:
                    if input[1779] < 1.2002537:
                        var11 = 0.15926872
                    else:
                        if input[903] >= 0.244607:
                            var11 = 0.20773731
                        else:
                            var11 = -0.08418658
    else:
        if input[4607] >= 0.15140015:
            if input[6785] >= 0.11443595:
                var11 = 0.07786731
            else:
                if input[4607] < 0.2991694:
                    if input[2957] >= 0.18397737:
                        var11 = -0.036184084
                    else:
                        var11 = -0.13743007
                else:
                    var11 = 0.053951394
        else:
            if input[2020] >= 0.17728917:
                if input[2020] < 0.22602095:
                    if input[8239] < 0.23995198:
                        var11 = -0.04403075
                    else:
                        var11 = -0.14739864
                else:
                    if input[2020] < 0.27830365:
                        var11 = 0.09284032
                    else:
                        var11 = -0.029750003
            else:
                if input[8626] >= 0.1311385:
                    if input[6785] >= 0.061546773:
                        if input[5736] < 0.33279613:
                            var11 = 0.1195758
                        else:
                            var11 = 0.00082858274
                    else:
                        if input[3844] < 0.22895801:
                            var11 = 0.077286
                        else:
                            var11 = -0.12506661
                else:
                    if input[9916] >= 0.13163704:
                        if input[2111] >= 0.12938312:
                            var11 = -0.14802514
                        else:
                            var11 = -0.020284211
                    else:
                        if input[2033] >= 0.19620371:
                            var11 = -0.1701326
                        else:
                            var11 = 0.030142963
    if input[6785] < 0.083166264:
        if input[4411] >= 0.1734649:
            var12 = -0.10534175
        else:
            if input[5280] >= 0.05525495:
                if input[5280] < 0.0737354:
                    if input[6785] < 0.06807458:
                        if input[5280] < 0.056576848:
                            var12 = 0.02151119
                        else:
                            var12 = 0.08178873
                    else:
                        if input[6785] < 0.07553596:
                            var12 = -0.0820856
                        else:
                            var12 = -0.012063315
                else:
                    var12 = 0.12685266
            else:
                if input[284] >= 0.1490245:
                    var12 = -0.078069635
                else:
                    if input[5779] >= 0.12893791:
                        var12 = -0.054274824
                    else:
                        if input[403] >= 0.07113375:
                            var12 = -0.051282108
                        else:
                            var12 = 0.12850855
    else:
        if input[1144] < 0.21387675:
            if input[3047] < 0.1694065:
                var12 = 0.22197703
            else:
                var12 = 0.14354426
        else:
            if input[5476] < 0.2921085:
                if input[6785] >= 0.10967751:
                    if input[3844] < 0.18338855:
                        var12 = -0.11466948
                    else:
                        if input[5476] < 0.23880072:
                            var12 = -0.009206123
                        else:
                            var12 = 0.059680123
                else:
                    if input[4666] >= 0.16947502:
                        var12 = -0.05356843
                    else:
                        if input[9614] >= 0.26416758:
                            var12 = -0.033807393
                        else:
                            var12 = 0.16434391
            else:
                if input[263] >= 0.16165602:
                    if input[268] < 0.5073344:
                        if input[5997] < 0.14402059:
                            var12 = 0.11563461
                        else:
                            var12 = 0.1943042
                    else:
                        if input[264] < 0.5886371:
                            var12 = 0.12548618
                        else:
                            var12 = -0.00024638936
                else:
                    if input[710] >= 0.21040188:
                        var12 = 0.13919167
                    else:
                        if input[3844] < 1.1679463:
                            var12 = 0.07968153
                        else:
                            var12 = -0.024132548
    if input[7224] < 0.29038507:
        if input[6785] < 0.1552051:
            if input[8239] >= 0.18935817:
                var13 = -0.008678349
            else:
                if input[3898] >= 0.22453628:
                    var13 = -0.006513403
                else:
                    if input[8626] >= 0.15756905:
                        if input[3667] < 0.45001036:
                            var13 = 0.06826993
                        else:
                            var13 = -0.020363715
                    else:
                        if input[1493] < 0.31545871:
                            var13 = 0.0118050845
                        else:
                            var13 = 0.11599822
        else:
            if input[7224] < 0.12369221:
                if input[8466] >= 0.10038626:
                    var13 = -0.025350993
                else:
                    var13 = 0.11856879
            else:
                if input[3641] >= 0.1348824:
                    if input[8626] < 0.2032789:
                        var13 = -0.045169894
                    else:
                        if input[2201] < 0.35301754:
                            var13 = -0.04056364
                        else:
                            var13 = 0.108349696
                else:
                    if input[1126] < 0.28528884:
                        if input[2658] < 0.22476609:
                            var13 = 0.0177075
                        else:
                            var13 = 0.102625266
                    else:
                        if input[9468] < 0.19219531:
                            var13 = 0.08489515
                        else:
                            var13 = -0.023670154
    else:
        if input[8239] < 0.082418025:
            if input[3609] >= 0.073660225:
                var13 = 0.03122202
            else:
                if input[1997] >= 0.0895276:
                    var13 = 0.022795629
                else:
                    if input[8239] < 0.08123742:
                        var13 = 0.15624237
                    else:
                        var13 = 0.054951847
        else:
            if input[5736] < 0.98390526:
                if input[8239] < 0.105920464:
                    var13 = 0.050820984
                else:
                    if input[5189] < 0.16194868:
                        var13 = 0.05548879
                    else:
                        if input[5113] >= 0.17815363:
                            var13 = 0.0510269
                        else:
                            var13 = 0.17090428
            else:
                if input[1144] < 0.21387675:
                    if input[3047] < 0.1694065:
                        var13 = 0.1972642
                    else:
                        var13 = 0.1356812
                else:
                    if input[9379] < 0.17845324:
                        if input[6321] < 0.23743644:
                            var13 = 0.16343747
                        else:
                            var13 = 0.061777383
                    else:
                        if input[710] >= 0.22767049:
                            var13 = 0.13346426
                        else:
                            var13 = -0.022873094
    if input[6785] < 0.083166264:
        if input[6785] < 0.06657117:
            if input[4941] >= 0.019123588:
                var14 = 0.016606417
            else:
                if input[4841] >= 0.022833884:
                    var14 = 0.018143514
                else:
                    if input[3898] >= 0.043379243:
                        var14 = 0.019244729
                    else:
                        if input[8626] >= 0.06703337:
                            var14 = 0.025338858
                        else:
                            var14 = 0.12948357
        else:
            if input[5280] < 0.0737354:
                if input[3916] >= 0.09318665:
                    var14 = -0.09637968
                else:
                    if input[6785] < 0.07553596:
                        if input[5280] < 0.06627581:
                            var14 = -0.014234729
                        else:
                            var14 = -0.0958706
                    else:
                        if input[9679] >= 0.08373093:
                            var14 = -0.055304416
                        else:
                            var14 = 0.05026585
            else:
                if input[4411] >= 0.1734649:
                    var14 = -0.1167589
                else:
                    if input[4170] < 0.12189623:
                        var14 = -0.08626086
                    else:
                        if input[5779] < 0.16162458:
                            var14 = -0.05316576
                        else:
                            var14 = 0.10887646
    else:
        if input[5378] < 1.001417:
            if input[3844] >= 0.19018213:
                var14 = -0.09942079
            else:
                if input[8867] >= 0.30845657:
                    var14 = -0.051158763
                else:
                    if input[3916] >= 0.14944507:
                        var14 = -0.0010778242
                    else:
                        if input[5496] >= 0.15984626:
                            var14 = 0.00919127
                        else:
                            var14 = 0.17841046
        else:
            if input[9679] < 0.09221593:
                if input[3916] >= 0.083009586:
                    if input[3916] < 0.097525634:
                        if input[3916] < 0.08860388:
                            var14 = -0.0035121073
                        else:
                            var14 = -0.07039342
                    else:
                        var14 = 0.03681794
                else:
                    if input[316] >= 0.046829153:
                        var14 = -0.07090544
                    else:
                        if input[5997] >= 0.079520315:
                            var14 = -0.007865128
                        else:
                            var14 = 0.15426685
            else:
                if input[5997] >= 0.13479984:
                    if input[9679] < 0.4148677:
                        if input[3916] < 0.15953882:
                            var14 = 0.022409268
                        else:
                            var14 = -0.11734831
                    else:
                        if input[5280] < 0.2012814:
                            var14 = -0.0728202
                        else:
                            var14 = 0.09150358
                else:
                    if input[5496] < 0.7983319:
                        if input[9526] < 0.4657235:
                            var14 = -0.096317105
                        else:
                            var14 = 0.10288312
                    else:
                        if input[5280] < 0.06378568:
                            var14 = 0.14918752
                        else:
                            var14 = -0.02317779
    if input[7224] < 0.29038507:
        if input[6785] < 0.20756286:
            if input[8385] >= 0.18800016:
                var15 = -0.019512173
            else:
                if input[3898] >= 0.23400651:
                    var15 = -0.014508861
                else:
                    if input[8626] >= 0.15756905:
                        if input[8626] < 0.2032789:
                            var15 = -0.012282043
                        else:
                            var15 = 0.07156368
                    else:
                        if input[7875] < 0.16802864:
                            var15 = 0.007971108
                        else:
                            var15 = 0.11336366
        else:
            if input[7224] < 0.12369221:
                if input[8466] >= 0.10038626:
                    var15 = -0.024110688
                else:
                    var15 = 0.115516536
            else:
                if input[3641] >= 0.1348824:
                    if input[2201] < 0.27155608:
                        var15 = -0.063528895
                    else:
                        if input[8070] >= 0.2397272:
                            var15 = -0.021509064
                        else:
                            var15 = 0.102786005
                else:
                    if input[1126] < 0.28528884:
                        if input[7224] < 0.12963822:
                            var15 = 0.015324141
                        else:
                            var15 = 0.10000628
                    else:
                        if input[9468] >= 0.14435217:
                            var15 = 0.08119072
                        else:
                            var15 = -0.029738659
    else:
        if input[5476] < 0.25431868:
            if input[6785] >= 0.10967751:
                if input[6878] < 0.38442856:
                    var15 = -0.107395366
                else:
                    if input[5476] < 0.23711544:
                        if input[5476] < 0.21874432:
                            var15 = 0.020907154
                        else:
                            var15 = -0.033212066
                    else:
                        var15 = 0.064236954
            else:
                if input[9468] >= 0.15708718:
                    var15 = -0.033892628
                else:
                    if input[3825] >= 0.16606335:
                        var15 = 0.0025936111
                    else:
                        if input[5491] >= 0.29747745:
                            var15 = -0.029507427
                        else:
                            var15 = 0.16554154
        else:
            if input[5736] < 0.98390526:
                if input[7141] < 0.15123974:
                    var15 = 0.03763864
                else:
                    if input[5189] < 0.16194868:
                        var15 = 0.046073057
                    else:
                        if input[5442] < 0.23923068:
                            var15 = 0.04686686
                        else:
                            var15 = 0.15886196
            else:
                if input[3844] < 0.19908191:
                    if input[3825] >= 0.1696696:
                        if input[3825] < 0.182618:
                            var15 = -0.1307335
                        else:
                            var15 = 0.008633687
                    else:
                        if input[9379] >= 0.10629529:
                            var15 = -0.020273041
                        else:
                            var15 = 0.11080425
                else:
                    if input[5280] < 0.05972259:
                        if input[6108] < 0.32275945:
                            var15 = -0.080892734
                        else:
                            var15 = 0.1455066
                    else:
                        if input[263] >= 0.16165602:
                            var15 = 0.13142988
                        else:
                            var15 = -0.021263612
    if input[4607] >= 0.15140015:
        if input[4607] < 0.21681306:
            if input[6785] >= 0.11443595:
                var16 = 0.060117323
            else:
                if input[5280] >= 0.09817982:
                    var16 = -0.028720101
                else:
                    var16 = -0.1297418
        else:
            if input[4607] < 0.28199577:
                if input[6785] < 0.1733538:
                    var16 = 0.035367776
                else:
                    var16 = -0.0492969
            else:
                var16 = 0.05824492
    else:
        if input[5280] >= 0.07766464:
            if input[4170] < 1.705911:
                if input[316] < 1.1247389:
                    if input[9213] < 0.19799241:
                        if input[316] < 0.17714117:
                            var16 = 0.12182181
                        else:
                            var16 = 0.033489536
                    else:
                        if input[1783] < 0.3843465:
                            var16 = 0.0884293
                        else:
                            var16 = -0.084814556
                else:
                    if input[360] < 0.92302686:
                        var16 = -0.11103147
                    else:
                        if input[2134] < 0.26709026:
                            var16 = -0.108334616
                        else:
                            var16 = 0.06735581
            else:
                if input[3000] < 0.357766:
                    if input[3480] >= 0.19949864:
                        var16 = -0.09699069
                    else:
                        if input[8453] < 0.23216556:
                            var16 = 0.14803687
                        else:
                            var16 = 0.043449983
                else:
                    if input[2221] >= 0.21026555:
                        var16 = 0.20886333
                    else:
                        if input[3371] < 0.3455014:
                            var16 = 0.1072618
                        else:
                            var16 = -0.06823637
        else:
            if input[2020] >= 0.17728917:
                if input[2020] < 0.22602095:
                    if input[8239] < 0.23995198:
                        var16 = -0.037444256
                    else:
                        var16 = -0.1363214
                else:
                    if input[2020] < 0.27830365:
                        var16 = 0.092131294
                    else:
                        var16 = -0.022643516
            else:
                if input[3916] >= 0.12404013:
                    if input[212] < 0.3113129:
                        var16 = -0.15035722
                    else:
                        if input[3844] >= 0.17259286:
                            var16 = 0.111049056
                        else:
                            var16 = -0.05542156
                else:
                    if input[8626] >= 0.1311385:
                        if input[6785] >= 0.061546773:
                            var16 = 0.011935809
                        else:
                            var16 = -0.10578888
                    else:
                        if input[2033] >= 0.19620371:
                            var16 = -0.15287285
                        else:
                            var16 = 0.029080702
    if input[8239] < 0.082418025:
        if input[3609] >= 0.09159884:
            var17 = 0.016248865
        else:
            if input[1997] >= 0.08125867:
                var17 = 0.021189716
            else:
                if input[8239] < 0.08123742:
                    var17 = 0.14442132
                else:
                    var17 = 0.047998227
    else:
        if input[7224] < 0.38224143:
            if input[6785] < 0.22547413:
                if input[8385] >= 0.18800016:
                    var17 = -0.020033969
                else:
                    if input[9379] >= 0.10025676:
                        if input[5113] < 0.35153866:
                            var17 = 0.07573918
                        else:
                            var17 = -0.0633508
                    else:
                        if input[8626] >= 0.15756905:
                            var17 = 0.023775455
                        else:
                            var17 = 0.11051839
            else:
                if input[7224] < 0.12369221:
                    if input[8466] >= 0.10038626:
                        var17 = -0.024404548
                    else:
                        var17 = 0.112012245
                else:
                    if input[3609] < 0.18332712:
                        if input[8626] < 0.15362759:
                            var17 = -0.033870008
                        else:
                            var17 = 0.097251505
                    else:
                        if input[9468] < 0.3867093:
                            var17 = 0.081986725
                        else:
                            var17 = -0.0082512535
        else:
            if input[1144] < 0.21387675:
                if input[3047] < 0.1694065:
                    var17 = 0.18064357
                else:
                    var17 = 0.1241073
            else:
                if input[9379] < 0.17845324:
                    if input[6321] < 0.23743644:
                        var17 = 0.1543868
                    else:
                        if input[2762] < 0.22394764:
                            var17 = 0.12102264
                        else:
                            var17 = 0.027596876
                else:
                    if input[710] < 0.67982006:
                        if input[710] < 0.21040188:
                            var17 = 0.03151157
                        else:
                            var17 = 0.12890601
                    else:
                        if input[5736] < 0.44086555:
                            var17 = 0.14624856
                        else:
                            var17 = -0.019363452
    if input[4607] >= 0.15140015:
        if input[4607] < 0.21681306:
            if input[6785] >= 0.11443595:
                var18 = 0.05740437
            else:
                if input[3047] < 0.2287076:
                    var18 = -0.02719861
                else:
                    var18 = -0.12550282
        else:
            if input[5280] < 0.16537653:
                var18 = -0.031166807
            else:
                if input[4607] < 0.2510862:
                    var18 = -0.026083535
                else:
                    var18 = 0.074722044
    else:
        if input[4012] >= 0.12833808:
            if input[6785] >= 0.08784678:
                if input[5280] >= 0.10292531:
                    var18 = -0.09906137
                else:
                    if input[6850] >= 0.30184093:
                        var18 = -0.05476088
                    else:
                        if input[8626] < 0.15823108:
                            var18 = -0.0712333
                        else:
                            var18 = 0.05739544
            else:
                if input[3844] < 0.20639126:
                    var18 = 0.08880716
                else:
                    if input[6611] < 1.2340925:
                        var18 = 0.116224766
                    else:
                        if input[7224] < 0.30779085:
                            var18 = 0.059224244
                        else:
                            var18 = -0.11039454
        else:
            if input[5280] >= 0.074496634:
                if input[4170] >= 0.12431107:
                    if input[316] >= 0.14765684:
                        if input[9213] < 0.19799241:
                            var18 = 0.09796274
                        else:
                            var18 = -0.07438583
                    else:
                        if input[360] < 0.92302686:
                            var18 = -0.1094258
                        else:
                            var18 = 0.06151132
                else:
                    if input[3000] < 0.2644257:
                        if input[3480] >= 0.19949864:
                            var18 = -0.08941551
                        else:
                            var18 = 0.08723145
                    else:
                        if input[962] < 0.51143813:
                            var18 = 0.15751228
                        else:
                            var18 = -0.056851458
            else:
                if input[8763] >= 0.15475793:
                    if input[6785] >= 0.09777701:
                        if input[6785] < 0.11077283:
                            var18 = -0.008900906
                        else:
                            var18 = 0.06641516
                    else:
                        if input[5496] >= 0.18952721:
                            var18 = 0.07922227
                        else:
                            var18 = -0.13543789
                else:
                    if input[9213] >= 0.16000241:
                        if input[2408] < 0.20140819:
                            var18 = -0.1465167
                        else:
                            var18 = -0.061444398
                    else:
                        if input[3916] >= 0.12404013:
                            var18 = -0.058412325
                        else:
                            var18 = 0.027644655
    if input[9679] < 0.08373093:
        if input[8239] >= 0.14873824:
            var19 = -0.08792346
        else:
            if input[5113] >= 0.100534074:
                var19 = 0.00544907
            else:
                if input[6785] >= 0.05811235:
                    if input[6785] < 0.07148315:
                        var19 = 0.014441669
                    else:
                        var19 = 0.078621596
                else:
                    if input[9341] >= 0.11765845:
                        var19 = 0.030258378
                    else:
                        var19 = 0.1460457
    else:
        if input[5378] < 1.001417:
            if input[3844] >= 0.19018213:
                var19 = -0.094519615
            else:
                if input[8867] >= 0.30845657:
                    var19 = -0.049380325
                else:
                    if input[3916] >= 0.14944507:
                        var19 = 0.015670074
                    else:
                        if input[5496] >= 0.15984626:
                            var19 = 0.0075337826
                        else:
                            var19 = 0.160103
        else:
            if input[3916] < 0.08860388:
                if input[5280] >= 0.064875074:
                    var19 = -0.04738794
                else:
                    if input[8239] >= 0.08123742:
                        var19 = 0.047191646
                    else:
                        if input[2965] < 0.1958744:
                            var19 = 0.031803917
                        else:
                            var19 = 0.16418122
            else:
                if input[263] >= 0.16165602:
                    if input[268] < 0.5073344:
                        if input[5997] < 0.14402059:
                            var19 = 0.105173826
                        else:
                            var19 = 0.16783552
                    else:
                        if input[264] < 0.5886371:
                            var19 = 0.11608292
                        else:
                            var19 = -0.017201427
                else:
                    if input[8466] < 0.26465392:
                        if input[5280] >= 0.06378568:
                            var19 = -0.045006406
                        else:
                            var19 = 0.074696876
                    else:
                        if input[7224] < 0.29038507:
                            var19 = 0.08401098
                        else:
                            var19 = -0.01995611
    if input[6785] < 0.06657117:
        if input[6252] >= 0.060243648:
            var20 = 0.016527094
        else:
            if input[4941] < 0.14609766:
                var20 = 0.015025103
            else:
                if input[3898] >= 0.024958584:
                    var20 = 0.017521627
                else:
                    if input[1997] >= 0.06402852:
                        var20 = 0.018353656
                    else:
                        if input[917] < 0.22280347:
                            var20 = 0.020167282
                        else:
                            var20 = 0.12528028
    else:
        if input[5496] < 0.4169824:
            if input[9526] < 0.4657235:
                var20 = -0.09363876
            else:
                if input[6785] >= 0.10946473:
                    if input[5496] < 0.20886993:
                        var20 = -0.038038164
                    else:
                        if input[5496] < 0.22405513:
                            var20 = 0.05603236
                        else:
                            var20 = 0.014464172
                else:
                    if input[8398] >= 0.1898279:
                        if input[5496] < 0.22088283:
                            var20 = -0.041248556
                        else:
                            var20 = 0.041509386
                    else:
                        if input[2957] >= 0.1645246:
                            var20 = -0.039416343
                        else:
                            var20 = 0.106588714
        else:
            if input[5476] < 0.29194674:
                if input[6785] >= 0.10967751:
                    if input[3844] < 0.18338855:
                        var20 = -0.104690984
                    else:
                        if input[6878] < 0.38682505:
                            var20 = -0.016887968
                        else:
                            var20 = 0.0489098
                else:
                    if input[4666] >= 0.16947502:
                        var20 = -0.0498674
                    else:
                        if input[5478] < 0.8049115:
                            var20 = -0.028554264
                        else:
                            var20 = 0.14029177
            else:
                if input[5997] >= 0.13479984:
                    if input[9679] >= 0.114979394:
                        if input[3916] < 0.15953882:
                            var20 = 0.029543942
                        else:
                            var20 = -0.12514581
                    else:
                        if input[8239] < 0.27070242:
                            var20 = -0.090511255
                        else:
                            var20 = 0.08196422
                else:
                    if input[5378] < 1.001417:
                        if input[3844] >= 0.15677217:
                            var20 = -0.024714481
                        else:
                            var20 = 0.14500745
                    else:
                        if input[5280] < 0.05972259:
                            var20 = 0.1342048
                        else:
                            var20 = -0.018332973
    if input[8239] < 0.08123742:
        if input[3609] >= 0.09159884:
            var21 = 0.011487586
        else:
            if input[1997] < 0.12916389:
                var21 = 0.023641719
            else:
                var21 = 0.13687997
    else:
        if input[4607] >= 0.15140015:
            if input[4607] < 0.21681306:
                if input[6785] >= 0.11443595:
                    var21 = 0.05595463
                else:
                    if input[3047] < 0.2287076:
                        var21 = -0.023687204
                    else:
                        var21 = -0.12142458
            else:
                if input[5280] < 0.16537653:
                    var21 = -0.02797454
                else:
                    if input[4607] < 0.2510862:
                        var21 = -0.023388866
                    else:
                        var21 = 0.071987726
        else:
            if input[4104] < 0.36291322:
                if input[8239] >= 0.09259192:
                    if input[5496] < 0.19161716:
                        var21 = -0.046687625
                    else:
                        if input[4104] < 0.19244204:
                            var21 = 0.0030303632
                        else:
                            var21 = 0.09424047
                else:
                    if input[2111] >= 0.113566846:
                        var21 = -0.041901417
                    else:
                        if input[8424] >= 0.14066409:
                            var21 = 0.038373012
                        else:
                            var21 = 0.14815408
            else:
                if input[5736] < 0.44086555:
                    if input[8239] < 0.105920464:
                        var21 = 0.035469376
                    else:
                        if input[3269] < 0.28820807:
                            var21 = 0.029484747
                        else:
                            var21 = 0.14329551
                else:
                    if input[263] < 0.2332693:
                        if input[5280] >= 0.092969336:
                            var21 = -0.09227301
                        else:
                            var21 = 0.12106271
                    else:
                        if input[7224] < 0.38224143:
                            var21 = 0.08033333
                        else:
                            var21 = -0.010573421
    if input[9679] < 0.09221593:
        if input[316] >= 0.12744412:
            var22 = -0.116216734
        else:
            if input[3916] >= 0.07751899:
                if input[8239] >= 0.08716159:
                    var22 = -0.10285338
                else:
                    if input[3916] < 0.09497498:
                        var22 = -0.027195333
                    else:
                        var22 = 0.039033554
            else:
                if input[6042] < 0.15386643:
                    var22 = -0.05753985
                else:
                    if input[284] >= 0.11749884:
                        var22 = -0.0052617444
                    else:
                        if input[6963] >= 0.12947512:
                            var22 = -0.007958116
                        else:
                            var22 = 0.13602452
    else:
        if input[710] >= 0.21040188:
            var22 = 0.1248906
        else:
            if input[6785] < 0.08898466:
                if input[4411] >= 0.18367273:
                    var22 = -0.12454152
                else:
                    if input[5280] < 0.07902782:
                        if input[6785] < 0.06807458:
                            var22 = 0.09441845
                        else:
                            var22 = -0.034261886
                    else:
                        if input[284] >= 0.14681634:
                            var22 = -0.10765313
                        else:
                            var22 = 0.108952336
            else:
                if input[9379] < 0.18057796:
                    if input[6321] < 0.23743644:
                        var22 = 0.14728959
                    else:
                        if input[9379] < 0.10629529:
                            var22 = 0.14515156
                        else:
                            var22 = 0.028136188
                else:
                    if input[3844] < 1.1679463:
                        if input[6785] >= 0.0974956:
                            var22 = -0.012207527
                        else:
                            var22 = 0.0842934
                    else:
                        if input[666] < 0.1821382:
                            var22 = 0.13925059
                        else:
                            var22 = -0.019199716
    if input[3916] < 0.083009586:
        if input[5025] >= 0.08001938:
            var23 = -0.0021642498
        else:
            if input[2965] >= 0.022774357:
                var23 = 0.02845174
            else:
                if input[6785] < 0.13519993:
                    var23 = 0.10444812
                else:
                    var23 = 0.16168979
    else:
        if input[5496] < 0.7983319:
            if input[9647] >= 0.26918325:
                var23 = -0.09943592
            else:
                if input[9526] < 0.4657235:
                    var23 = -0.088933714
                else:
                    if input[76] < 0.3013982:
                        var23 = -0.07357429
                    else:
                        if input[6785] >= 0.10946473:
                            var23 = 0.003712466
                        else:
                            var23 = 0.097597
        else:
            if input[5476] < 1.171566:
                if input[6785] >= 0.10967751:
                    if input[6878] < 0.38442856:
                        var23 = -0.09844988
                    else:
                        if input[3844] < 0.20392661:
                            var23 = -0.01613237
                        else:
                            var23 = 0.04328164
                else:
                    if input[5494] < 1.0953032:
                        var23 = -0.10150617
                    else:
                        if input[3825] >= 0.16606335:
                            var23 = -0.02151236
                        else:
                            var23 = 0.12616971
            else:
                if input[5736] < 0.98390526:
                    if input[9576] < 0.21851471:
                        var23 = 0.025199238
                    else:
                        if input[5018] < 0.2069436:
                            var23 = 0.026468625
                        else:
                            var23 = 0.13505323
                else:
                    if input[5997] >= 0.13479984:
                        if input[5997] < 0.20598951:
                            var23 = 0.07143561
                        else:
                            var23 = -0.10616185
                    else:
                        if input[7224] < 0.38224143:
                            var23 = 0.07664442
                        else:
                            var23 = -0.017274937
    if input[4607] >= 0.15140015:
        if input[4607] < 0.21681306:
            if input[6785] >= 0.11443595:
                var24 = 0.05445229
            else:
                if input[3047] < 0.2287076:
                    var24 = -0.020923128
                else:
                    var24 = -0.11804892
        else:
            if input[5280] < 0.16537653:
                var24 = -0.025034485
            else:
                if input[4607] < 0.2510862:
                    var24 = -0.022000823
                else:
                    var24 = 0.0688265
    else:
        if input[4012] >= 0.12833808:
            if input[6785] >= 0.08784678:
                if input[5280] >= 0.10292531:
                    var24 = -0.0933624
                else:
                    if input[6850] >= 0.30184093:
                        var24 = -0.050123334
                    else:
                        if input[8626] < 0.15823108:
                            var24 = -0.07115757
                        else:
                            var24 = 0.056030184
            else:
                if input[3844] < 0.20639126:
                    if input[4012] < 0.15378574:
                        if input[3844] < 0.17332034:
                            var24 = 0.07237988
                        else:
                            var24 = -0.01748055
                    else:
                        var24 = 0.1001001
                else:
                    if input[6611] < 1.2340925:
                        var24 = 0.11294816
                    else:
                        if input[9916] < 0.20859876:
                            var24 = -0.13938208
                        else:
                            var24 = -0.08283178
        else:
            if input[8626] >= 0.15267338:
                if input[2474] < 0.5256529:
                    var24 = -0.14017203
                else:
                    if input[2628] < 0.40155745:
                        var24 = 0.18448095
                    else:
                        if input[6785] >= 0.12869906:
                            var24 = 0.0076060644
                        else:
                            var24 = -0.08063578
            else:
                if input[8763] >= 0.17155485:
                    if input[3609] >= 0.13922729:
                        var24 = -0.13833076
                    else:
                        if input[6785] >= 0.10778583:
                            var24 = 0.034527477
                        else:
                            var24 = -0.07496066
                else:
                    if input[5280] >= 0.10129063:
                        if input[903] < 0.63816714:
                            var24 = 0.16893741
                        else:
                            var24 = -0.036677726
                    else:
                        if input[5869] >= 0.21954967:
                            var24 = -0.13671713
                        else:
                            var24 = 0.019709911
    if input[8239] < 0.08123742:
        if input[3609] >= 0.09159884:
            var25 = 0.0064168433
        else:
            if input[6785] >= 0.05811235:
                var25 = 0.031831723
            else:
                var25 = 0.13226297
    else:
        if input[710] >= 0.21040188:
            var25 = 0.12104243
        else:
            if input[6978] < 0.35193887:
                var25 = 0.14950792
            else:
                if input[6785] < 0.06657117:
                    if input[3916] >= 0.062188055:
                        var25 = 0.012516486
                    else:
                        if input[9916] >= 0.076358944:
                            var25 = 0.020333797
                        else:
                            var25 = 0.11805625
                else:
                    if input[2344] < 0.2415315:
                        if input[248] < 0.21890488:
                            var25 = 0.1401742
                        else:
                            var25 = 0.04783162
                    else:
                        if input[3844] < 0.19908191:
                            var25 = 0.07797921
                        else:
                            var25 = -0.013079099
    if input[2020] >= 0.17652127:
        if input[2020] < 0.22602095:
            if input[8239] < 0.23995198:
                var26 = -0.032311123
            else:
                var26 = -0.12483578
        else:
            if input[2020] < 0.27830365:
                var26 = 0.093883775
            else:
                if input[2020] < 0.33079714:
                    var26 = -0.029539919
                else:
                    var26 = 0.014546432
    else:
        if input[4607] >= 0.15140015:
            if input[4607] < 0.21681306:
                if input[6785] >= 0.11443595:
                    var26 = 0.050837394
                else:
                    if input[3047] < 0.2287076:
                        var26 = -0.019365773
                    else:
                        var26 = -0.11558665
            else:
                if input[4607] < 0.28199577:
                    if input[4607] < 0.23690724:
                        var26 = 0.032467976
                    else:
                        var26 = -0.040621385
                else:
                    var26 = 0.048402447
        else:
            if input[8626] >= 0.1311385:
                if input[484] < 0.33438396:
                    var26 = -0.13405243
                else:
                    if input[2628] < 0.40155745:
                        var26 = 0.18187475
                    else:
                        if input[6785] >= 0.061546773:
                            var26 = -0.0021524527
                        else:
                            var26 = -0.06794887
            else:
                if input[2033] >= 0.19620371:
                    if input[809] < 0.19113877:
                        var26 = -0.13677128
                    else:
                        var26 = -0.040167592
                else:
                    if input[3916] >= 0.12042419:
                        if input[9916] >= 0.16038296:
                            var26 = -0.12891786
                        else:
                            var26 = -0.033088345
                    else:
                        if input[5280] >= 0.10180071:
                            var26 = -0.03067275
                        else:
                            var26 = 0.019672858
    if input[5280] < 0.05972259:
        if input[6108] < 0.32275945:
            var27 = -0.083951436
        else:
            if input[6252] >= 0.091475256:
                var27 = -0.03068443
            else:
                if input[5531] >= 0.07530933:
                    var27 = 0.021349648
                else:
                    var27 = 0.1352519
    else:
        if input[2677] < 2.00001:
            if input[9679] >= 0.13621159:
                var27 = -0.007515061
            else:
                if input[8466] < 0.11503485:
                    var27 = 0.025315983
                else:
                    if input[5531] < 0.35453176:
                        var27 = 0.02704358
                    else:
                        if input[5189] < 0.22996761:
                            var27 = 0.06003023
                        else:
                            var27 = 0.20195977
        else:
            if input[4104] < 0.27949:
                if input[5496] >= 0.13898797:
                    if input[8239] < 0.12939683:
                        var27 = -0.050483257
                    else:
                        if input[8424] < 0.23078826:
                            var27 = -0.01505235
                        else:
                            var27 = 0.0915177
                else:
                    if input[5378] >= 0.21852799:
                        var27 = 0.02217231
                    else:
                        if input[8239] >= 0.09259192:
                            var27 = 0.044197563
                        else:
                            var27 = 0.14255969
            else:
                if input[263] >= 0.16165602:
                    if input[268] < 0.5073344:
                        var27 = 0.14651148
                    else:
                        if input[264] < 0.5886371:
                            var27 = 0.10844535
                        else:
                            var27 = -0.010118063
                else:
                    if input[7224] < 0.19207613:
                        if input[3311] >= 0.16207545:
                            var27 = -0.0778838
                        else:
                            var27 = 0.08380195
                    else:
                        if input[5736] < 0.44086555:
                            var27 = 0.1265301
                        else:
                            var27 = -0.012206382
    if input[9679] < 0.08373093:
        if input[8239] >= 0.14873824:
            var28 = -0.08123418
        else:
            if input[5113] >= 0.100534074:
                var28 = -0.0075661573
            else:
                if input[6785] >= 0.05811235:
                    if input[6785] < 0.07148315:
                        var28 = -0.0039444924
                    else:
                        var28 = 0.06956463
                else:
                    if input[7081] < 0.16598743:
                        var28 = 0.033644434
                    else:
                        var28 = 0.13427135
    else:
        if input[8466] < 0.09015203:
            var28 = 0.15887357
        else:
            if input[710] >= 0.21040188:
                var28 = 0.117395096
            else:
                if input[5378] < 1.001417:
                    if input[3844] >= 0.19018213:
                        var28 = -0.09455082
                    else:
                        if input[8867] >= 0.30845657:
                            var28 = -0.0566666
                        else:
                            var28 = 0.12794451
                else:
                    if input[7972] < 0.24531172:
                        if input[8378] < 0.33069387:
                            var28 = -0.11041059
                        else:
                            var28 = 0.12233821
                    else:
                        if input[9386] < 0.22473142:
                            var28 = 0.13398431
                        else:
                            var28 = -0.010964741
    if input[3916] < 0.083009586:
        if input[5025] >= 0.08001938:
            var29 = -0.003621564
        else:
            if input[2965] < 0.24718189:
                var29 = 0.029473802
            else:
                var29 = 0.14608116
    else:
        if input[5476] < 0.25431868:
            if input[6785] >= 0.10967751:
                if input[3844] < 0.18338855:
                    var29 = -0.10022532
                else:
                    if input[5476] < 0.23880072:
                        if input[5476] < 0.21874432:
                            var29 = 0.05319762
                        else:
                            var29 = -0.054776996
                    else:
                        var29 = 0.05819679
            else:
                if input[5478] >= 0.30021042:
                    var29 = -0.04415947
                else:
                    if input[5491] >= 0.29747745:
                        var29 = -0.042805098
                    else:
                        if input[9468] >= 0.15708718:
                            var29 = -0.034809005
                        else:
                            var29 = 0.12931646
        else:
            if input[1144] < 0.21387675:
                var29 = 0.13833296
            else:
                if input[9468] < 0.22134733:
                    if input[8178] < 0.37111157:
                        var29 = -0.13735893
                    else:
                        if input[8239] >= 0.10109547:
                            var29 = -0.012070372
                        else:
                            var29 = 0.09421272
                else:
                    if input[6326] < 0.30980593:
                        if input[6328] >= 0.28239128:
                            var29 = 0.01700591
                        else:
                            var29 = 0.16503105
                    else:
                        if input[5997] >= 0.13479984:
                            var29 = 0.058826752
                        else:
                            var29 = -0.013212298
    if input[8239] < 0.08123742:
        if input[3609] >= 0.09159884:
            var30 = 0.0019070872
        else:
            if input[6785] >= 0.05811235:
                var30 = 0.026553014
            else:
                var30 = 0.1276884
    else:
        if input[903] < 1.5411538:
            if input[4241] >= 0.18709259:
                var30 = -0.037391644
            else:
                if input[873] < 0.30677226:
                    var30 = -0.012145932
                else:
                    if input[3916] < 0.12645009:
                        var30 = 0.08271284
                    else:
                        if input[5819] >= 0.14204787:
                            var30 = 0.052116107
                        else:
                            var30 = 0.18526772
        else:
            if input[5496] < 0.49882984:
                if input[9647] >= 0.26918325:
                    var30 = -0.09224802
                else:
                    if input[9526] < 0.4657235:
                        var30 = -0.08604475
                    else:
                        if input[5496] < 0.13140815:
                            var30 = 0.12502907
                        else:
                            var30 = 0.07362208
            else:
                if input[3000] < 0.2626688:
                    if input[284] < 0.3204352:
                        var30 = -0.11077114
                    else:
                        if input[4170] >= 0.1178779:
                            var30 = -0.10603607
                        else:
                            var30 = 0.11259957
                else:
                    if input[3844] < 0.20342515:
                        if input[3825] >= 0.16606335:
                            var30 = -0.0621634
                        else:
                            var30 = 0.085696295
                    else:
                        if input[263] >= 0.16165602:
                            var30 = 0.10595836
                        else:
                            var30 = -0.0127620315
    if input[2020] >= 0.17652127:
        if input[2020] < 0.22602095:
            if input[8239] < 0.23995198:
                var31 = -0.028488064
            else:
                var31 = -0.11973529
        else:
            if input[2020] < 0.27830365:
                var31 = 0.08937549
            else:
                if input[2020] < 0.33079714:
                    var31 = -0.025848968
                else:
                    var31 = 0.016461449
    else:
        if input[4607] >= 0.15140015:
            if input[4607] < 0.21681306:
                if input[6785] >= 0.11443595:
                    var31 = 0.04789122
                else:
                    if input[3047] < 0.2287076:
                        var31 = -0.017254256
                    else:
                        var31 = -0.11287457
            else:
                if input[4607] < 0.28199577:
                    if input[4607] < 0.23690724:
                        var31 = 0.033046212
                    else:
                        var31 = -0.036331143
                else:
                    var31 = 0.048453387
        else:
            if input[5531] >= 0.11662267:
                if input[6785] >= 0.10619637:
                    if input[5280] >= 0.10129063:
                        var31 = -0.10522367
                    else:
                        if input[9679] < 0.123514034:
                            var31 = -0.022198325
                        else:
                            var31 = 0.10170158
                else:
                    if input[5280] < 0.12051032:
                        if input[5694] < 0.3021742:
                            var31 = 0.15495291
                        else:
                            var31 = -0.022701185
                    else:
                        if input[5736] < 0.24399836:
                            var31 = 0.09998996
                        else:
                            var31 = -0.10163398
            else:
                if input[9274] >= 0.21646078:
                    var31 = -0.13238326
                else:
                    if input[8626] >= 0.1311385:
                        if input[484] < 0.33438396:
                            var31 = -0.12765636
                        else:
                            var31 = -0.03922498
                    else:
                        if input[2111] >= 0.111870445:
                            var31 = -0.052373122
                        else:
                            var31 = 0.015360609
    if input[5280] < 0.05972259:
        if input[6108] < 0.32275945:
            var32 = -0.07755891
        else:
            if input[4819] >= 0.02819953:
                var32 = -0.03549715
            else:
                if input[371] >= 0.11008394:
                    var32 = 0.01927856
                else:
                    var32 = 0.12993257
    else:
        if input[2677] < 2.00001:
            if input[5189] < 0.23776038:
                if input[9679] >= 0.090631:
                    var32 = -0.013119563
                else:
                    var32 = 0.025730217
            else:
                if input[8466] < 0.11503485:
                    var32 = 0.01743077
                else:
                    if input[5531] >= 0.14282015:
                        var32 = 0.02336262
                    else:
                        if input[9679] >= 0.11966396:
                            var32 = 0.07030938
                        else:
                            var32 = 0.18887518
        else:
            if input[710] >= 0.21040188:
                var32 = 0.114811376
            else:
                if input[6611] < 0.29233888:
                    if input[3825] < 0.1726117:
                        var32 = -0.025297964
                    else:
                        if input[8416] >= 0.3512492:
                            var32 = -0.03491011
                        else:
                            var32 = 0.15481031
                else:
                    if input[5736] < 0.44086555:
                        if input[8239] < 0.105920464:
                            var32 = 0.04411729
                        else:
                            var32 = 0.12480774
                    else:
                        if input[6418] < 0.34574562:
                            var32 = 0.122715175
                        else:
                            var32 = -0.009630699
    if input[6785] < 0.06657117:
        if input[1997] < 0.1253682:
            var33 = 0.012818553
        else:
            if input[917] < 0.22280347:
                var33 = 0.0139378505
            else:
                if input[9916] >= 0.076358944:
                    var33 = 0.018912232
                else:
                    if input[3916] >= 0.062188055:
                        var33 = 0.018694205
                    else:
                        if input[3432] >= 0.070851855:
                            var33 = 0.017236128
                        else:
                            var33 = 0.11685456
    else:
        if input[248] < 0.21890488:
            if input[248] < 0.17922641:
                if input[248] < 0.13620862:
                    var33 = 0.07612022
                else:
                    var33 = -0.045772623
            else:
                var33 = 0.12937401
        else:
            if input[5476] < 1.171566:
                if input[6785] >= 0.10967751:
                    if input[3844] < 0.18338855:
                        var33 = -0.09737187
                    else:
                        if input[3844] < 0.21133515:
                            var33 = 0.060913246
                        else:
                            var33 = -0.00045320715
                else:
                    if input[5494] < 1.0953032:
                        var33 = -0.09653543
                    else:
                        if input[3825] >= 0.16606335:
                            var33 = -0.029381305
                        else:
                            var33 = 0.10754895
            else:
                if input[6978] < 0.35193887:
                    var33 = 0.13644621
                else:
                    if input[5378] < 1.001417:
                        if input[3844] >= 0.15677217:
                            var33 = -0.04173887
                        else:
                            var33 = 0.11631987
                    else:
                        if input[8887] >= 0.17582916:
                            var33 = 0.098557286
                        else:
                            var33 = -0.010709702
    if input[9679] < 0.08373093:
        if input[8239] >= 0.14873824:
            var34 = -0.075134434
        else:
            if input[5113] >= 0.100534074:
                var34 = -0.007575825
            else:
                if input[6785] >= 0.05811235:
                    var34 = 0.034173876
                else:
                    if input[7081] < 0.16598743:
                        var34 = 0.027100062
                    else:
                        var34 = 0.12945776
    else:
        if input[8466] < 0.09015203:
            var34 = 0.15040146
        else:
            if input[710] >= 0.21040188:
                var34 = 0.11248125
            else:
                if input[2680] < 2.00001:
                    if input[2680] < 0.25381976:
                        if input[2680] < 0.23268394:
                            var34 = 0.15579818
                        else:
                            var34 = -0.0072444226
                    else:
                        var34 = 0.21741512
                else:
                    if input[7972] < 0.23276408:
                        if input[1131] < 0.29302365:
                            var34 = -0.07538477
                        else:
                            var34 = 0.11497714
                    else:
                        if input[5736] < 0.98390526:
                            var34 = 0.11836686
                        else:
                            var34 = -0.0087947585
    var35 = var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19 + var20 + var21 + var22 + var23 + var24 + var25 + var26 + var27 + var28 + var29 + var30 + var31 + var32 + var33 + var34
    if input[3916] < 0.083009586:
        if input[5025] >= 0.08001938:
            var36 = -0.0033665167
        else:
            if input[5189] >= 0.01505754:
                var36 = 0.031755056
            else:
                var36 = 0.13960822
    else:
        if input[8792] >= 0.13564949:
            if input[6785] >= 0.106665775:
                var36 = 0.03114433
            else:
                if input[2957] < 0.21118052:
                    var36 = -0.020817017
                else:
                    if input[5280] >= 0.10954209:
                        var36 = -0.028906291
                    else:
                        var36 = -0.11659255
        else:
            if input[9213] >= 0.1443012:
                if input[809] < 0.19113877:
                    var36 = -0.1304742
                else:
                    if input[5105] < 0.24098027:
                        var36 = -0.10941111
                    else:
                        if input[4594] < 0.3230736:
                            var36 = 0.14295375
                        else:
                            var36 = -0.029714158
            else:
                if input[2722] >= 0.18979347:
                    var36 = -0.1309139
                else:
                    if input[8763] >= 0.16829471:
                        if input[6785] >= 0.11240005:
                            var36 = 0.049250465
                        else:
                            var36 = -0.100580476
                    else:
                        if input[5869] >= 0.21954967:
                            var36 = -0.12385609
                        else:
                            var36 = 0.0073190304
    if input[8239] < 0.08123742:
        if input[3609] >= 0.09159884:
            var37 = 0.00048438748
        else:
            if input[6785] >= 0.05811235:
                var37 = 0.019440696
            else:
                var37 = 0.123861544
    else:
        if input[903] < 1.5411538:
            if input[4241] >= 0.18709259:
                var37 = -0.036322303
            else:
                if input[873] < 0.30677226:
                    var37 = -0.010302594
                else:
                    if input[3916] < 0.12645009:
                        if input[3916] < 0.10404498:
                            var37 = 0.09200799
                        else:
                            var37 = 0.025265744
                    else:
                        var37 = 0.16640362
        else:
            if input[6326] < 0.30980593:
                if input[9468] >= 0.1556249:
                    var37 = -0.053909104
                else:
                    if input[6328] >= 0.28239128:
                        if input[5378] < 0.2431983:
                            var37 = -0.026520712
                        else:
                            var37 = 0.04674412
                    else:
                        if input[3844] >= 0.14586645:
                            var37 = 0.061759826
                        else:
                            var37 = 0.16068466
            else:
                if input[7224] < 0.19207613:
                    if input[6785] < 0.20756286:
                        if input[3898] >= 0.22453628:
                            var37 = -0.025092706
                        else:
                            var37 = 0.09869847
                    else:
                        if input[8626] >= 0.10695634:
                            var37 = -0.07199586
                        else:
                            var37 = 0.056227364
                else:
                    if input[5496] < 0.7983319:
                        if input[5496] < 0.13140815:
                            var37 = 0.120752394
                        else:
                            var37 = 0.060994215
                    else:
                        if input[4770] < 0.21458688:
                            var37 = 0.11554646
                        else:
                            var37 = -0.010284368
    if input[5280] < 0.06627581:
        if input[284] >= 0.11749884:
            if input[284] < 0.19039635:
                var38 = -0.00983028
            else:
                var38 = -0.09778147
        else:
            if input[6078] >= 0.034522075:
                var38 = -0.08397337
            else:
                if input[3047] >= 0.09803413:
                    var38 = -0.06033152
                else:
                    if input[4574] >= 0.07949263:
                        var38 = -0.03177329
                    else:
                        if input[842] >= 0.04275238:
                            var38 = -0.0205085
                        else:
                            var38 = 0.12482909
    else:
        if input[9468] < 0.1372231:
            if input[3311] >= 0.13562259:
                var38 = -0.049123548
            else:
                if input[9307] >= 0.13289702:
                    var38 = -0.031079603
                else:
                    if input[2429] >= 0.061490063:
                        var38 = -0.0087285815
                    else:
                        if input[5113] >= 0.11603851:
                            var38 = 0.036624562
                        else:
                            var38 = 0.13855194
        else:
            if input[3844] < 1.1679463:
                if input[6785] >= 0.0974956:
                    if input[3825] < 0.27538994:
                        if input[3825] < 0.18823664:
                            var38 = -0.04105859
                        else:
                            var38 = -0.12538284
                    else:
                        if input[3875] < 1.0078187:
                            var38 = -0.1156523
                        else:
                            var38 = 0.010356255
                else:
                    if input[3844] < 0.12264583:
                        if input[9379] >= 0.06036637:
                            var38 = 0.019068098
                        else:
                            var38 = 0.119052224
                    else:
                        if input[1812] >= 0.17552367:
                            var38 = -0.117446624
                        else:
                            var38 = 0.059173744
            else:
                if input[6785] < 0.3671209:
                    if input[3825] >= 0.15518488:
                        if input[4012] >= 0.14790934:
                            var38 = 0.08786927
                        else:
                            var38 = -0.11299113
                    else:
                        if input[5280] < 0.22088154:
                            var38 = -0.02721009
                        else:
                            var38 = 0.044409167
                else:
                    if input[5997] >= 0.13479984:
                        if input[8239] >= 0.12774764:
                            var38 = -0.119951665
                        else:
                            var38 = 0.06806918
                    else:
                        if input[710] >= 0.21040188:
                            var38 = 0.113852255
                        else:
                            var38 = -0.019119298
    if input[2020] >= 0.17652127:
        if input[2020] < 0.22602095:
            if input[8239] < 0.23995198:
                var39 = -0.025972743
            else:
                var39 = -0.11427664
        else:
            if input[2020] < 0.27830365:
                var39 = 0.08549028
            else:
                if input[2020] < 0.33079714:
                    var39 = -0.020487105
                else:
                    var39 = 0.018391011
    else:
        if input[4607] >= 0.15140015:
            if input[4607] < 0.21681306:
                if input[6785] >= 0.11240005:
                    var39 = 0.034161415
                else:
                    if input[5280] >= 0.09746591:
                        var39 = -0.02116201
                    else:
                        var39 = -0.10952532
            else:
                if input[4607] < 0.28199577:
                    if input[4607] < 0.23690724:
                        var39 = 0.033081833
                    else:
                        var39 = -0.033482853
                else:
                    var39 = 0.04720255
        else:
            if input[263] < 0.23401643:
                if input[5280] >= 0.092969336:
                    var39 = -0.08516001
                else:
                    if input[8466] >= 0.10793656:
                        var39 = -0.06973844
                    else:
                        if input[268] < 0.5073344:
                            var39 = 0.1332634
                        else:
                            var39 = 0.08171113
            else:
                if input[2677] < 2.00001:
                    if input[9679] >= 0.13621159:
                        var39 = -0.0057286085
                    else:
                        if input[8466] < 0.11503485:
                            var39 = 0.009297001
                        else:
                            var39 = 0.16269241
                else:
                    if input[3000] < 0.38271728:
                        if input[284] < 0.3204352:
                            var39 = -0.083577186
                        else:
                            var39 = 0.09250741
                    else:
                        if input[1144] < 0.21387675:
                            var39 = 0.12898216
                        else:
                            var39 = -0.0030165024
    if input[9679] < 0.09221593:
        if input[316] >= 0.12744412:
            var40 = -0.105722
        else:
            if input[3916] >= 0.07751899:
                if input[8239] >= 0.08716159:
                    var40 = -0.091709994
                else:
                    if input[3916] < 0.09497498:
                        var40 = -0.031675998
                    else:
                        var40 = 0.034197114
            else:
                if input[8626] >= 0.08068065:
                    if input[9679] < 0.08207893:
                        var40 = -0.018522508
                    else:
                        var40 = 0.0029606088
                else:
                    if input[3000] >= 0.13377078:
                        var40 = -0.02328959
                    else:
                        if input[284] >= 0.11749884:
                            var40 = -0.015287618
                        else:
                            var40 = 0.1236242
    else:
        if input[3916] < 0.09318665:
            if input[284] < 0.3704383:
                var40 = -0.064686544
            else:
                if input[8239] >= 0.08123742:
                    if input[3916] < 0.08571792:
                        var40 = 0.040623643
                    else:
                        var40 = -0.03159445
                else:
                    if input[3609] >= 0.088108934:
                        var40 = -0.012796344
                    else:
                        if input[4012] >= 0.085529275:
                            var40 = 0.0020452964
                        else:
                            var40 = 0.14237052
        else:
            if input[7334] < 0.53020924:
                if input[3371] >= 0.21087503:
                    if input[3371] < 0.27114743:
                        var40 = -0.08630686
                    else:
                        var40 = 0.01811049
                else:
                    if input[5280] >= 0.11687931:
                        if input[4170] >= 0.20658599:
                            var40 = 0.037989076
                        else:
                            var40 = -0.11372938
                    else:
                        if input[1997] >= 0.17189217:
                            var40 = -0.07968953
                        else:
                            var40 = 0.13116464
            else:
                if input[8466] < 0.26465392:
                    if input[5280] >= 0.06378568:
                        if input[3000] >= 0.15320171:
                            var40 = 0.08195575
                        else:
                            var40 = -0.04315065
                    else:
                        if input[8239] >= 0.09214643:
                            var40 = -0.02275682
                        else:
                            var40 = 0.068538904
                else:
                    if input[8887] < 0.25115293:
                        if input[263] < 0.19672017:
                            var40 = 0.1286346
                        else:
                            var40 = 0.05311447
                    else:
                        if input[7972] < 0.37534994:
                            var40 = 0.100921355
                        else:
                            var40 = -0.01271893
    if input[6611] < 0.29233888:
        if input[3825] < 0.1726117:
            var41 = -0.027845576
        else:
            if input[8416] >= 0.3512492:
                var41 = -0.03768198
            else:
                if input[2144] < 0.61254185:
                    var41 = -0.022834754
                else:
                    if input[9679] >= 0.22241081:
                        var41 = 0.009370109
                    else:
                        if input[5819] >= 0.14648575:
                            var41 = 0.017577438
                        else:
                            var41 = 0.16090848
    else:
        if input[4104] < 0.30212373:
            if input[8239] >= 0.09259192:
                if input[5496] < 0.19161716:
                    var41 = -0.05286118
                else:
                    if input[4104] < 0.19244204:
                        var41 = -0.007556533
                    else:
                        var41 = 0.08133681
            else:
                if input[2111] >= 0.113566846:
                    var41 = -0.039149642
                else:
                    if input[5496] >= 0.22720504:
                        var41 = -0.022695424
                    else:
                        if input[5378] >= 0.1904471:
                            var41 = 0.03319339
                        else:
                            var41 = 0.1296678
        else:
            if input[6418] < 0.34574562:
                if input[70] >= 0.23681445:
                    var41 = -0.094441675
                else:
                    if input[3214] >= 0.24440584:
                        var41 = -0.056134906
                    else:
                        if input[7509] >= 0.2082361:
                            var41 = -0.056459557
                        else:
                            var41 = 0.14404823
            else:
                if input[6785] < 0.06657117:
                    if input[4666] >= 0.08242442:
                        var41 = 0.011760801
                    else:
                        if input[3916] >= 0.05963973:
                            var41 = 0.012098438
                        else:
                            var41 = 0.108892374
                else:
                    if input[2344] < 0.2415315:
                        if input[248] < 0.21890488:
                            var41 = 0.12366692
                        else:
                            var41 = 0.042834785
                    else:
                        if input[2680] < 2.00001:
                            var41 = 0.15952097
                        else:
                            var41 = -0.008082411
    if input[5476] < 0.25431868:
        if input[6785] >= 0.10967751:
            if input[6878] < 0.38442856:
                var42 = -0.09108075
            else:
                if input[5476] < 0.23711544:
                    if input[5476] < 0.21874432:
                        var42 = 0.021078711
                    else:
                        var42 = -0.036409754
                else:
                    var42 = 0.051680665
        else:
            if input[8317] < 0.40402943:
                var42 = -0.046128433
            else:
                if input[5478] >= 0.30021042:
                    var42 = -0.0477414
                else:
                    if input[3825] >= 0.16606335:
                        var42 = -0.01258339
                    else:
                        if input[5531] >= 0.12654969:
                            var42 = -0.020576539
                        else:
                            var42 = 0.12098815
    else:
        if input[5280] < 0.05525495:
            if input[5280] < 0.048058297:
                var42 = 0.123018935
            else:
                var42 = 0.057353146
        else:
            if input[7224] < 0.19207613:
                if input[6785] < 0.20756286:
                    if input[8239] >= 0.14084679:
                        var42 = -0.031563632
                    else:
                        if input[3898] >= 0.22453628:
                            var42 = -0.025061687
                        else:
                            var42 = 0.098996624
                else:
                    if input[8626] >= 0.10695634:
                        if input[3609] < 0.136824:
                            var42 = -0.0019941425
                        else:
                            var42 = -0.09043759
                    else:
                        if input[8466] >= 0.10038626:
                            var42 = -0.073098585
                        else:
                            var42 = 0.06319894
            else:
                if input[5496] < 0.3194097:
                    if input[5496] < 0.13140815:
                        var42 = 0.11648109
                    else:
                        if input[3916] < 0.121714525:
                            var42 = -0.0965381
                        else:
                            var42 = 0.064114854
                else:
                    if input[3844] < 0.39837793:
                        if input[6785] >= 0.0974956:
                            var42 = -0.0127967615
                        else:
                            var42 = 0.06698897
                    else:
                        if input[5378] < 1.001417:
                            var42 = 0.11043457
                        else:
                            var42 = -0.011792277
    if input[3371] < 0.20181295:
        if input[7376] >= 0.18533817:
            var43 = -0.09667062
        else:
            if input[2957] < 0.16803464:
                var43 = -0.05937575
            else:
                if input[8239] >= 0.10258515:
                    if input[8239] < 0.11457161:
                        if input[3371] < 0.17540549:
                            var43 = 0.0018759392
                        else:
                            var43 = -0.073283635
                    else:
                        if input[8239] < 0.17770505:
                            var43 = 0.12499948
                        else:
                            var43 = 0.034620777
                else:
                    if input[3087] >= 0.13693716:
                        var43 = -0.02340387
                    else:
                        if input[6785] >= 0.09920907:
                            var43 = 0.024859194
                        else:
                            var43 = 0.16400145
    else:
        if input[3269] < 0.20988214:
            if input[316] < 0.19915296:
                var43 = -0.10434059
            else:
                if input[3087] < 0.17757325:
                    var43 = -0.064111166
                else:
                    if input[9916] >= 0.12331891:
                        var43 = -0.01887305
                    else:
                        if input[9379] >= 0.1450157:
                            var43 = -0.002864552
                        else:
                            var43 = 0.11780993
        else:
            if input[710] >= 0.21040188:
                var43 = 0.10851791
            else:
                if input[5736] < 0.44086555:
                    if input[8239] < 0.105920464:
                        var43 = 0.02287739
                    else:
                        if input[6534] >= 0.20772035:
                            var43 = 0.013923237
                        else:
                            var43 = 0.1188468
                else:
                    if input[8239] < 0.083582796:
                        if input[8626] >= 0.08068065:
                            var43 = -0.04211776
                        else:
                            var43 = 0.11536695
                    else:
                        if input[9386] < 0.22473142:
                            var43 = 0.12056979
                        else:
                            var43 = -0.008206201
    if input[903] < 1.5411538:
        if input[4241] >= 0.18709259:
            var44 = -0.036094066
        else:
            if input[873] < 0.30677226:
                var44 = -0.008052895
            else:
                if input[6785] >= 0.0974956:
                    if input[2221] < 0.4159861:
                        var44 = 0.10381754
                    else:
                        var44 = -0.0030626038
                else:
                    if input[2429] < 0.24549001:
                        var44 = 0.044864967
                    else:
                        if input[8626] < 0.12685595:
                            var44 = 0.049054604
                        else:
                            var44 = 0.16175535
    else:
        if input[6404] < 1.1448104:
            if input[6785] < 0.073931284:
                var44 = 0.063548304
            else:
                var44 = 0.21310115
        else:
            if input[8466] < 0.09015203:
                if input[8361] >= 0.07340768:
                    var44 = 0.032877512
                else:
                    var44 = 0.14125341
            else:
                if input[6326] < 0.8345217:
                    if input[9468] >= 0.1556249:
                        var44 = -0.060607024
                    else:
                        if input[3844] >= 0.19908191:
                            var44 = -0.053864457
                        else:
                            var44 = 0.13321453
                else:
                    if input[666] < 0.1821382:
                        if input[8239] >= 0.10258515:
                            var44 = -0.08837838
                        else:
                            var44 = 0.11227192
                    else:
                        if input[8626] >= 0.11628212:
                            var44 = -0.05706159
                        else:
                            var44 = -0.0022021674
    if input[2033] < 0.3023872:
        var45 = -0.12153856
    else:
        if input[4012] >= 0.12520215:
            if input[6785] >= 0.08784678:
                if input[5280] >= 0.10292531:
                    var45 = -0.08902837
                else:
                    if input[3110] >= 0.18830264:
                        var45 = 0.12448356
                    else:
                        if input[8626] < 0.19056751:
                            var45 = -0.092355914
                        else:
                            var45 = 0.035742108
            else:
                if input[4666] < 0.14174043:
                    var45 = -0.12545793
                else:
                    if input[5997] >= 0.13028292:
                        if input[5997] < 0.13479984:
                            var45 = -0.10433332
                        else:
                            var45 = -0.17329441
                    else:
                        if input[3504] < 1.4315313:
                            var45 = 0.13975649
                        else:
                            var45 = -0.039927464
        else:
            if input[4607] >= 0.15140015:
                if input[4607] < 0.21681306:
                    if input[6785] < 0.13024926:
                        var45 = 0.015944917
                    else:
                        if input[5280] < 0.11065827:
                            var45 = -0.018920233
                        else:
                            var45 = -0.107610844
                else:
                    if input[4607] < 0.28199577:
                        if input[4607] < 0.23690724:
                            var45 = 0.030924056
                        else:
                            var45 = -0.029894695
                    else:
                        var45 = 0.04691601
            else:
                if input[5531] >= 0.11662267:
                    if input[6785] >= 0.10619637:
                        if input[5280] >= 0.10129063:
                            var45 = -0.08521838
                        else:
                            var45 = 0.08301
                    else:
                        if input[5280] < 0.12051032:
                            var45 = -0.003534258
                        else:
                            var45 = -0.08624459
                else:
                    if input[9274] >= 0.21646078:
                        var45 = -0.12194254
                    else:
                        if input[6306] < 0.30334857:
                            var45 = -0.1217375
                        else:
                            var45 = 0.010899426
    if input[9525] < 1.241941:
        if input[6785] < 0.073931284:
            var46 = 0.07621588
        else:
            var46 = 0.2033766
    else:
        if input[3000] < 0.2626688:
            if input[4170] >= 0.1178779:
                var46 = -0.10483148
            else:
                if input[3480] >= 0.19949864:
                    var46 = -0.08555504
                else:
                    if input[2310] >= 0.19519673:
                        var46 = -0.09368127
                    else:
                        if input[7390] >= 0.18412404:
                            var46 = -0.07991668
                        else:
                            var46 = 0.105079435
        else:
            if input[5476] < 1.171566:
                if input[6785] >= 0.10967751:
                    if input[3844] < 0.18338855:
                        var46 = -0.08857218
                    else:
                        if input[3844] < 0.21133515:
                            var46 = 0.05817805
                        else:
                            var46 = -0.00034662255
                else:
                    if input[5494] < 1.0953032:
                        var46 = -0.09118661
                    else:
                        if input[8466] >= 0.13605368:
                            var46 = -0.02509361
                        else:
                            var46 = 0.096468315
            else:
                if input[710] >= 0.21040188:
                    var46 = 0.10627582
                else:
                    if input[9468] < 0.23291637:
                        if input[8178] < 0.37111157:
                            var46 = -0.13178913
                        else:
                            var46 = 0.06569842
                    else:
                        if input[7334] < 0.43812683:
                            var46 = 0.08088461
                        else:
                            var46 = -0.009661558
    if input[2677] < 2.00001:
        if input[5189] >= 0.17032836:
            var47 = -0.0057241544
        else:
            if input[8466] < 0.11503485:
                var47 = 0.0010394765
            else:
                if input[5531] >= 0.14282015:
                    var47 = 0.018338665
                else:
                    if input[9679] >= 0.11966396:
                        if input[2677] < 0.2687319:
                            var47 = -0.00016068168
                        else:
                            var47 = 0.08982553
                    else:
                        var47 = 0.16255777
    else:
        if input[2679] < 1.4071828:
            if input[5531] >= 0.15608986:
                var47 = -0.06336603
            else:
                if input[3609] < 0.16402929:
                    var47 = -0.01188274
                else:
                    if input[5189] >= 0.18266423:
                        var47 = 0.022160422
                    else:
                        var47 = 0.1789693
        else:
            if input[6611] < 0.29233888:
                if input[3825] < 0.1726117:
                    var47 = -0.0301348
                else:
                    if input[8416] >= 0.3512492:
                        var47 = -0.036434542
                    else:
                        if input[2144] < 0.61254185:
                            var47 = -0.021813354
                        else:
                            var47 = 0.14319418
            else:
                if input[5280] < 0.05525495:
                    if input[5280] < 0.048058297:
                        var47 = 0.119262934
                    else:
                        var47 = 0.05195586
                else:
                    if input[5189] < 1.1187681:
                        if input[4666] >= 0.12500043:
                            var47 = -0.048870623
                        else:
                            var47 = 0.08812329
                    else:
                        if input[5496] < 0.7983319:
                            var47 = 0.066482425
                        else:
                            var47 = -0.008149175
    if input[4068] >= 0.15810794:
        if input[5280] >= 0.10080631:
            if input[4068] < 0.25262994:
                var48 = 0.026235288
            else:
                var48 = -0.050435957
        else:
            if input[4076] < 0.29521683:
                var48 = -0.00020343527
            else:
                if input[6785] >= 0.11370214:
                    var48 = 0.020181425
                else:
                    if input[8626] >= 0.1428273:
                        var48 = 0.011682539
                    else:
                        if input[7168] < 0.22236083:
                            var48 = -0.027740572
                        else:
                            var48 = -0.14387897
    else:
        if input[2020] >= 0.17652127:
            if input[2020] < 0.22602095:
                if input[8239] < 0.23995198:
                    var48 = -0.024235867
                else:
                    var48 = -0.10993804
            else:
                if input[2020] < 0.27830365:
                    var48 = 0.08319907
                else:
                    if input[2020] < 0.33079714:
                        var48 = -0.01680791
                    else:
                        var48 = 0.0201462
        else:
            if input[5869] >= 0.21954967:
                var48 = -0.11870978
            else:
                if input[2839] >= 0.22153445:
                    var48 = -0.11756756
                else:
                    if input[4607] >= 0.15140015:
                        if input[4607] < 0.21681306:
                            var48 = -0.10107603
                        else:
                            var48 = 0.026712775
                    else:
                        if input[2722] >= 0.18979347:
                            var48 = -0.120499864
                        else:
                            var48 = 0.0075775445
    if input[9679] < 0.08373093:
        if input[8239] >= 0.14873824:
            var49 = -0.07101321
        else:
            if input[5113] >= 0.0999666:
                var49 = -0.015063937
            else:
                if input[6785] >= 0.05811235:
                    var49 = 0.019725256
                else:
                    if input[8424] >= 0.0798584:
                        var49 = 0.027914325
                    else:
                        var49 = 0.12233362
    else:
        if input[7972] < 0.23276408:
            if input[9679] >= 0.1408579:
                if input[8239] < 0.19487584:
                    var49 = -0.10946777
                else:
                    var49 = 0.038228963
            else:
                if input[8466] >= 0.11098771:
                    var49 = -0.04828658
                else:
                    if input[8626] >= 0.11517362:
                        var49 = -0.013077614
                    else:
                        if input[8239] >= 0.13646792:
                            var49 = 0.0032035436
                        else:
                            var49 = 0.122881666
        else:
            if input[3269] < 0.20942095:
                if input[316] < 0.15524016:
                    var49 = -0.09660374
                else:
                    if input[3087] < 0.17757325:
                        var49 = -0.06348343
                    else:
                        if input[5280] >= 0.09583368:
                            var49 = -0.03922609
                        else:
                            var49 = 0.106449
            else:
                if input[263] >= 0.16165602:
                    if input[5280] < 0.14344557:
                        var49 = -0.07865242
                    else:
                        if input[263] < 0.26686603:
                            var49 = 0.103545524
                        else:
                            var49 = -0.05946436
                else:
                    if input[9006] < 0.6959612:
                        var49 = 0.12351679
                    else:
                        if input[5997] >= 0.14948702:
                            var49 = -0.15437022
                        else:
                            var49 = -0.005257004
    if input[8239] < 0.07834077:
        if input[8466] >= 0.071994476:
            var50 = 0.009438488
        else:
            var50 = 0.11843659
    else:
        if input[6978] < 0.35193887:
            var50 = 0.12410255
        else:
            if input[6785] < 0.06657117:
                if input[9307] >= 0.06479331:
                    var50 = 0.01036081
                else:
                    if input[3432] >= 0.06536444:
                        var50 = 0.012147824
                    else:
                        if input[3916] >= 0.046011634:
                            var50 = 0.01492689
                        else:
                            var50 = 0.1105478
            else:
                if input[2344] < 0.2415315:
                    if input[248] < 0.21890488:
                        var50 = 0.11790772
                    else:
                        var50 = 0.04031915
                else:
                    if input[903] < 1.5411538:
                        if input[4241] >= 0.18709259:
                            var50 = -0.03500805
                        else:
                            var50 = 0.13391832
                    else:
                        if input[6326] < 0.30980593:
                            var50 = 0.1150882
                        else:
                            var50 = -0.006671205
    if input[3916] < 0.083009586:
        if input[5025] >= 0.08001938:
            var51 = -0.0022862393
        else:
            if input[5189] < 0.11507707:
                var51 = 0.028134724
            else:
                var51 = 0.13016818
    else:
        if input[5736] < 0.44086555:
            if input[8239] < 0.105920464:
                var51 = 0.021872768
            else:
                if input[5189] < 0.17679358:
                    var51 = 0.012044072
                else:
                    if input[2876] >= 0.2429167:
                        var51 = 0.014751586
                    else:
                        if input[5113] >= 0.16126575:
                            var51 = 0.023492966
                        else:
                            var51 = 0.12021132
        else:
            if input[2680] < 2.00001:
                if input[8398] >= 0.16102141:
                    var51 = 0.020005263
                else:
                    if input[8466] >= 0.11442392:
                        var51 = 0.025458654
                    else:
                        if input[6785] < 0.105886854:
                            var51 = 0.07734122
                        else:
                            var51 = 0.17238872
            else:
                if input[5459] < 0.2726077:
                    if input[8466] >= 0.13314956:
                        var51 = -0.004027835
                    else:
                        if input[5496] < 0.15757152:
                            var51 = 0.025828708
                        else:
                            var51 = 0.16049771
                else:
                    if input[6465] < 0.5095433:
                        if input[9468] >= 0.1796428:
                            var51 = -0.026422767
                        else:
                            var51 = 0.14724527
                    else:
                        if input[7224] < 1.2077464:
                            var51 = 0.060231067
                        else:
                            var51 = -0.0068837753
    if input[8923] >= 0.17807269:
        if input[7224] >= 0.13196553:
            var52 = 0.024100287
        else:
            if input[4012] >= 0.1563553:
                var52 = 0.021655207
            else:
                if input[5587] < 0.85903096:
                    var52 = -0.020402607
                else:
                    if input[6785] < 0.114644684:
                        var52 = -0.03902269
                    else:
                        if input[5280] < 0.10740642:
                            var52 = -0.038913578
                        else:
                            var52 = -0.1490023
    else:
        if input[6730] < 0.55022043:
            if input[5898] >= 0.1443223:
                var52 = -0.06374488
            else:
                if input[8466] >= 0.13682023:
                    var52 = 0.0124155525
                else:
                    if input[3844] < 0.19761482:
                        var52 = 0.03408454
                    else:
                        if input[9379] < 0.13763148:
                            var52 = 0.040414046
                        else:
                            var52 = 0.17861287
        else:
            if input[4104] < 0.27949:
                if input[8424] >= 0.13968639:
                    if input[76] < 0.26661566:
                        var52 = -0.07827078
                    else:
                        if input[5496] >= 0.17378882:
                            var52 = -0.022637825
                        else:
                            var52 = 0.0661426
                else:
                    if input[8239] < 0.114009485:
                        if input[4104] < 0.15444608:
                            var52 = 0.07236941
                        else:
                            var52 = -0.08390824
                    else:
                        if input[5177] < 0.34091845:
                            var52 = -0.0028217686
                        else:
                            var52 = 0.1202564
            else:
                if input[6418] < 0.34574562:
                    if input[70] >= 0.23681445:
                        var52 = -0.08855207
                    else:
                        if input[3214] >= 0.24440584:
                            var52 = -0.038073596
                        else:
                            var52 = 0.12380358
                else:
                    if input[3371] < 0.20181295:
                        if input[7376] >= 0.18533817:
                            var52 = -0.09155676
                        else:
                            var52 = 0.11378404
                    else:
                        if input[6611] < 0.29233888:
                            var52 = 0.118543826
                        else:
                            var52 = -0.0046281824
    if input[8466] < 0.09015203:
        if input[8361] >= 0.06947656:
            var53 = 0.030715672
        else:
            var53 = 0.1350909
    else:
        if input[710] >= 0.21040188:
            var53 = 0.10475762
        else:
            if input[8887] >= 0.17582916:
                if input[8887] < 0.25115293:
                    if input[5113] < 0.19800337:
                        var53 = -0.07196703
                    else:
                        if input[8239] < 0.13819507:
                            var53 = -0.049359225
                        else:
                            var53 = 0.10185975
                else:
                    var53 = -0.088978976
            else:
                if input[6785] < 0.3671209:
                    if input[3825] >= 0.15518488:
                        if input[4012] >= 0.14347032:
                            var53 = 0.07300191
                        else:
                            var53 = -0.10793135
                    else:
                        if input[5280] < 0.22088154:
                            var53 = -0.0175495
                        else:
                            var53 = 0.037874088
                else:
                    if input[3844] < 0.49723622:
                        if input[9679] >= 0.088964455:
                            var53 = -0.0059805596
                        else:
                            var53 = 0.06775099
                    else:
                        if input[3609] >= 0.14809255:
                            var53 = -0.09093039
                        else:
                            var53 = -0.009412961
    if input[1779] >= 0.042298205:
        if input[1781] >= 0.3292537:
            var54 = 0.049742445
        else:
            if input[5280] < 0.06627581:
                var54 = 0.07319557
            else:
                var54 = 0.27161586
    else:
        if input[4068] >= 0.15810794:
            if input[4076] >= 0.27483052:
                if input[4068] < 0.21971458:
                    var54 = -0.053784158
                else:
                    var54 = 0.054065447
            else:
                if input[5280] >= 0.11019604:
                    if input[4068] < 0.26972327:
                        var54 = 0.02914663
                    else:
                        var54 = -0.03017342
                else:
                    if input[6785] >= 0.11370214:
                        var54 = 0.017653437
                    else:
                        if input[8626] >= 0.1428273:
                            var54 = 0.01214347
                        else:
                            var54 = -0.13547379
        else:
            if input[8398] < 0.12470183:
                if input[284] >= 0.040838227:
                    var54 = 0.013535631
                else:
                    if input[5113] >= 0.043592546:
                        var54 = 0.03573897
                    else:
                        var54 = 0.15307067
            else:
                if input[9006] < 0.6959612:
                    var54 = 0.12085922
                else:
                    if input[666] < 0.1821382:
                        if input[8466] < 0.115878716:
                            var54 = -0.08292331
                        else:
                            var54 = 0.107611
                    else:
                        if input[3000] < 0.38271728:
                            var54 = 0.07321334
                        else:
                            var54 = -0.0040073968
    if input[2677] < 2.00001:
        if input[5189] >= 0.17032836:
            var55 = -0.0063615493
        else:
            if input[8239] < 0.10301286:
                var55 = 0.00875493
            else:
                if input[5531] >= 0.14282015:
                    var55 = 0.016588097
                else:
                    if input[9679] >= 0.11966396:
                        if input[2677] < 0.2687319:
                            var55 = -0.0008244157
                        else:
                            var55 = 0.08407008
                    else:
                        if input[4012] >= 0.14217967:
                            var55 = 0.044326134
                        else:
                            var55 = 0.15457538
    else:
        if input[7972] < 0.23276408:
            if input[9679] >= 0.1408579:
                if input[8239] < 0.19487584:
                    var55 = -0.10273627
                else:
                    var55 = 0.034948308
            else:
                if input[8466] >= 0.11098771:
                    var55 = -0.043976586
                else:
                    if input[8626] < 0.15942161:
                        var55 = -0.012794534
                    else:
                        if input[8239] >= 0.13646792:
                            var55 = 0.005326837
                        else:
                            var55 = 0.11885393
        else:
            if input[5476] < 0.25431868:
                if input[6785] >= 0.10967751:
                    if input[6878] < 0.38442856:
                        var55 = -0.08507421
                    else:
                        if input[5476] < 0.23711544:
                            var55 = -0.0070993598
                        else:
                            var55 = 0.046999056
                else:
                    if input[5491] >= 0.29747745:
                        var55 = -0.050926875
                    else:
                        if input[5478] >= 0.30021042:
                            var55 = -0.04614563
                        else:
                            var55 = 0.10176685
            else:
                if input[248] < 0.21890488:
                    if input[248] < 0.17922641:
                        var55 = 0.0062465016
                    else:
                        var55 = 0.1153124
                else:
                    if input[5280] < 0.06627581:
                        if input[284] >= 0.11749884:
                            var55 = -0.0635327
                        else:
                            var55 = 0.09589427
                    else:
                        if input[5378] < 1.001417:
                            var55 = 0.08794307
                        else:
                            var55 = -0.0074545457
    if input[6404] < 1.1448104:
        var56 = 0.18067198
    else:
        if input[5736] < 0.44086555:
            if input[9468] < 0.14319532:
                if input[5736] < 0.15973069:
                    var56 = 0.06337895
                else:
                    var56 = -0.031731635
            else:
                if input[5189] < 0.19776435:
                    var56 = 0.012919153
                else:
                    if input[6534] >= 0.20238593:
                        var56 = 0.015078082
                    else:
                        if input[5113] >= 0.16126575:
                            var56 = 0.022049032
                        else:
                            var56 = 0.117964104
        else:
            if input[3916] >= 0.10326767:
                if input[3844] >= 0.17259286:
                    var56 = 0.10643961
                else:
                    if input[6785] >= 0.0825758:
                        if input[8626] < 0.13241:
                            var56 = -0.10262147
                        else:
                            var56 = 0.026158376
                    else:
                        if input[212] < 0.3113129:
                            var56 = -0.115411
                        else:
                            var56 = -0.04433688
            else:
                if input[2020] >= 0.17728917:
                    if input[2020] < 0.22602095:
                        var56 = -0.107061125
                    else:
                        if input[2020] < 0.27704504:
                            var56 = 0.09304288
                        else:
                            var56 = -0.018043926
                else:
                    if input[6306] < 0.60670716:
                        var56 = -0.11765192
                    else:
                        if input[2033] < 0.3023872:
                            var56 = -0.11408533
                        else:
                            var56 = 0.0054429155
    if input[5422] >= 0.19210783:
        if input[1347] < 0.46965322:
            var57 = 0.032756504
        else:
            if input[8239] < 0.10807894:
                var57 = -0.02289795
            else:
                if input[8626] < 0.15267338:
                    var57 = -0.028727258
                else:
                    if input[4438] < 0.27826312:
                        var57 = -0.03958944
                    else:
                        if input[1812] >= 0.23204167:
                            var57 = -0.04058614
                        else:
                            var57 = -0.15911268
    else:
        if input[7141] >= 0.13388465:
            if input[7152] >= 0.28120142:
                if input[7152] < 0.3683485:
                    var57 = 0.13703233
                else:
                    var57 = 0.026507095
            else:
                if input[6785] >= 0.09469094:
                    if input[9679] < 0.16624108:
                        var57 = -0.059489083
                    else:
                        if input[3609] < 0.1524959:
                            var57 = 0.12382915
                        else:
                            var57 = 0.029701326
                else:
                    if input[7160] >= 0.26938942:
                        if input[9679] < 0.12926725:
                            var57 = 0.11287371
                        else:
                            var57 = 0.015857534
                    else:
                        if input[4170] < 0.22175354:
                            var57 = 0.028471148
                        else:
                            var57 = -0.1046251
        else:
            if input[9274] >= 0.21646078:
                var57 = -0.11592951
            else:
                if input[8923] >= 0.17807269:
                    if input[7224] >= 0.13196553:
                        var57 = 0.023290548
                    else:
                        if input[4012] >= 0.1563553:
                            var57 = 0.02130743
                        else:
                            var57 = -0.12669112
                else:
                    if input[2722] >= 0.18979347:
                        var57 = -0.116539
                    else:
                        if input[6306] < 0.60670716:
                            var57 = -0.11522736
                        else:
                            var57 = 0.006625221
    if input[2679] < 1.4071828:
        if input[5531] >= 0.15608986:
            var58 = -0.0608001
        else:
            if input[3609] < 0.16402929:
                var58 = -0.010599094
            else:
                if input[5189] >= 0.18266423:
                    var58 = 0.020231284
                else:
                    var58 = 0.16151208
    else:
        if input[8239] < 0.07834077:
            if input[6785] >= 0.056591045:
                var58 = 0.008941039
            else:
                var58 = 0.11602455
        else:
            if input[710] >= 0.22767049:
                var58 = 0.102899864
            else:
                if input[3641] < 0.2786712:
                    if input[2201] < 0.2552148:
                        var58 = -0.05669687
                    else:
                        if input[8626] >= 0.12920874:
                            var58 = -0.0196046
                        else:
                            var58 = 0.09935868
                else:
                    if input[3609] >= 0.10058274:
                        if input[6785] < 0.2709589:
                            var58 = 0.041708954
                        else:
                            var58 = -0.081418484
                    else:
                        if input[8887] >= 0.17582916:
                            var58 = 0.08476943
                        else:
                            var58 = -0.0031074537
    if input[9679] < 0.08373093:
        if input[8239] >= 0.14873824:
            var59 = -0.06765566
        else:
            if input[5113] >= 0.0999666:
                var59 = -0.015693523
            else:
                if input[6785] >= 0.05811235:
                    var59 = 0.012199544
                else:
                    if input[9307] >= 0.078318916:
                        var59 = 0.030090956
                    else:
                        var59 = 0.11958623
    else:
        if input[962] < 0.51143813:
            var59 = 0.18051401
        else:
            if input[903] < 1.5411538:
                if input[4170] >= 0.14478311:
                    if input[4170] < 0.14731102:
                        var59 = -0.092521496
                    else:
                        if input[9679] < 0.35507712:
                            var59 = 0.006951594
                        else:
                            var59 = 0.110289134
                else:
                    if input[8626] < 0.13203143:
                        var59 = 0.026737288
                    else:
                        if input[6785] >= 0.0974956:
                            var59 = 0.04327067
                        else:
                            var59 = 0.1513785
            else:
                if input[9525] < 1.241941:
                    var59 = 0.16974826
                else:
                    if input[1350] < 0.28775734:
                        if input[6785] < 0.1337869:
                            var59 = -0.08528249
                        else:
                            var59 = 0.11656544
                    else:
                        if input[9006] < 0.6959612:
                            var59 = 0.11776686
                        else:
                            var59 = -0.005272054
    if input[1316] < 0.41825908:
        var60 = 0.18323368
    else:
        if input[3269] < 0.20942095:
            if input[316] < 0.15524016:
                var60 = -0.09143263
            else:
                if input[3087] < 0.17757325:
                    var60 = -0.06231978
                else:
                    if input[3371] >= 0.1997008:
                        var60 = -0.039391946
                    else:
                        if input[9916] < 0.16446052:
                            var60 = -0.038377482
                        else:
                            var60 = 0.10423304
        else:
            if input[7334] < 0.2070433:
                if input[7376] >= 0.19209282:
                    var60 = -0.06956271
                else:
                    if input[3916] >= 0.10253184:
                        var60 = -0.023401326
                    else:
                        if input[5819] < 0.1639485:
                            var60 = 0.0008131691
                        else:
                            var60 = 0.13378617
            else:
                if input[9386] < 0.5192439:
                    var60 = 0.112462215
                else:
                    if input[9468] < 0.1372231:
                        if input[3311] >= 0.13562259:
                            var60 = -0.051126886
                        else:
                            var60 = 0.11002584
                    else:
                        if input[263] >= 0.16165602:
                            var60 = 0.08467867
                        else:
                            var60 = -0.0062028463
    if input[3916] < 0.083009586:
        if input[5025] >= 0.08001938:
            var61 = -0.00057883654
        else:
            if input[5189] < 0.11507707:
                var61 = 0.020706108
            else:
                var61 = 0.12585488
    else:
        if input[5990] < 1.0:
            if input[8239] >= 0.10071536:
                var61 = 0.04966405
            else:
                var61 = 0.16661286
        else:
            if input[1664] < 0.22088127:
                if input[6785] >= 0.08784678:
                    var61 = -0.042889204
                else:
                    if input[8466] < 0.20375818:
                        var61 = 0.019231565
                    else:
                        var61 = 0.11764808
            else:
                if input[7972] < 0.23276408:
                    if input[9679] >= 0.1408579:
                        if input[8239] < 0.19487584:
                            var61 = -0.097172946
                        else:
                            var61 = 0.026057998
                    else:
                        if input[1812] >= 0.20486079:
                            var61 = -0.06891609
                        else:
                            var61 = 0.10574277
                else:
                    if input[6730] < 0.55022043:
                        if input[5898] >= 0.1443223:
                            var61 = -0.060260665
                        else:
                            var61 = 0.14291847
                    else:
                        if input[5496] < 0.7983319:
                            var61 = 0.059805114
                        else:
                            var61 = -0.0061980146
    if input[2680] < 2.00001:
        if input[2680] < 0.25381976:
            if input[2680] < 0.23268394:
                if input[8239] < 0.10258515:
                    var62 = 0.03653149
                else:
                    var62 = 0.12742399
            else:
                if input[6785] < 0.10881177:
                    var62 = 0.010951852
                else:
                    var62 = -0.054415412
        else:
            var62 = 0.1646391
    else:
        if input[5736] < 0.44086555:
            if input[9468] < 0.14319532:
                if input[5736] < 0.15973069:
                    var62 = 0.058200903
                else:
                    var62 = -0.03057523
            else:
                if input[5189] < 0.2070526:
                    var62 = 0.012655786
                else:
                    if input[5113] >= 0.17151158:
                        var62 = 0.02189751
                    else:
                        var62 = 0.11592171
        else:
            if input[6418] < 0.34574562:
                if input[70] >= 0.23681445:
                    var62 = -0.08267962
                else:
                    if input[3214] >= 0.24440584:
                        var62 = -0.053432968
                    else:
                        if input[7081] >= 0.16598743:
                            var62 = -0.05859024
                        else:
                            var62 = 0.12226438
            else:
                if input[5476] < 1.171566:
                    if input[3844] >= 0.16551208:
                        if input[3844] < 0.17332034:
                            var62 = -0.11594619
                        else:
                            var62 = 0.012507148
                    else:
                        if input[5494] < 1.0953032:
                            var62 = -0.09201624
                        else:
                            var62 = 0.080856346
                else:
                    if input[5280] < 0.048058297:
                        var62 = 0.11466793
                    else:
                        if input[3641] < 0.2786712:
                            var62 = 0.0834947
                        else:
                            var62 = -0.006700014
    if input[963] < 0.7767438:
        if input[4170] >= 0.11331123:
            var63 = -0.035094358
        else:
            var63 = 0.17445052
    else:
        if input[5459] < 0.2726077:
            if input[6465] >= 0.21177806:
                var63 = -0.008974141
            else:
                if input[5496] >= 0.15686233:
                    var63 = 0.024893723
                else:
                    if input[9379] < 0.1387586:
                        var63 = 0.043681372
                    else:
                        var63 = 0.15945117
        else:
            if input[6465] < 0.5095433:
                if input[9468] >= 0.1796428:
                    var63 = -0.027753938
                else:
                    if input[6785] >= 0.14634067:
                        var63 = -0.0036069432
                    else:
                        if input[9916] < 0.13619913:
                            var63 = -0.0019657803
                        else:
                            var63 = 0.14829682
            else:
                if input[710] >= 0.21040188:
                    var63 = 0.101388685
                else:
                    if input[2677] < 2.00001:
                        if input[5189] < 0.23776038:
                            var63 = -0.0076205814
                        else:
                            var63 = 0.12466474
                    else:
                        if input[8398] < 0.12470183:
                            var63 = 0.13647018
                        else:
                            var63 = -0.004872083
    if input[5869] >= 0.21954967:
        var64 = -0.11291212
    else:
        if input[8792] >= 0.13564949:
            if input[6785] >= 0.10455085:
                var64 = 0.018582266
            else:
                if input[2957] < 0.24613264:
                    var64 = -0.019651629
                else:
                    if input[5280] >= 0.09583368:
                        var64 = -0.02281267
                    else:
                        var64 = -0.10812049
        else:
            if input[5531] >= 0.11662267:
                if input[6785] >= 0.10619637:
                    if input[5280] >= 0.10129063:
                        var64 = -0.09920018
                    else:
                        if input[4666] < 0.17961295:
                            var64 = -0.02890029
                        else:
                            var64 = 0.08125835
                else:
                    if input[5694] >= 0.20176665:
                        var64 = 0.20749144
                    else:
                        if input[5280] < 0.11656167:
                            var64 = -0.007698515
                        else:
                            var64 = -0.07602342
            else:
                if input[9213] >= 0.16659303:
                    if input[6785] < 0.32187775:
                        if input[5280] < 0.10766058:
                            var64 = 0.11700713
                        else:
                            var64 = 0.008938721
                    else:
                        if input[3844] >= 0.20639126:
                            var64 = 0.1051088
                        else:
                            var64 = -0.09622516
                else:
                    if input[5422] >= 0.17566824:
                        if input[1347] < 0.46965322:
                            var64 = 0.032029927
                        else:
                            var64 = -0.12581386
                    else:
                        if input[2722] >= 0.18979347:
                            var64 = -0.11279708
                        else:
                            var64 = 0.0076807174
    if input[6785] < 0.06657117:
        if input[6785] < 0.05536623:
            if input[7224] >= 0.060239978:
                var65 = 0.00617449
            else:
                var65 = 0.109672956
        else:
            if input[6785] < 0.056591045:
                var65 = -0.023728987
            else:
                if input[5280] < 0.05972259:
                    var65 = 0.0065837577
                else:
                    if input[8239] < 0.21218416:
                        var65 = 0.015826585
                    else:
                        var65 = 0.09119487
    else:
        if input[9307] < 0.11202816:
            if input[9916] >= 0.076358944:
                var65 = 0.003378357
            else:
                if input[8239] >= 0.057607286:
                    var65 = 0.041106075
                else:
                    var65 = 0.15342356
        else:
            if input[1779] < 1.2002537:
                if input[1781] >= 0.3292537:
                    var65 = 0.047582723
                else:
                    var65 = 0.22586122
            else:
                if input[9006] < 0.6959612:
                    var65 = 0.11542572
                else:
                    if input[2344] < 0.2415315:
                        if input[248] < 0.21890488:
                            var65 = 0.111474834
                        else:
                            var65 = 0.0379001
                    else:
                        if input[3000] < 0.38271728:
                            var65 = 0.06540509
                        else:
                            var65 = -0.005450565
    if input[6404] < 1.1448104:
        var66 = 0.16623694
    else:
        if input[5372] >= 0.24515349:
            var66 = 0.23766477
        else:
            if input[4068] >= 0.15810794:
                if input[4076] >= 0.27483052:
                    if input[4068] < 0.21971458:
                        var66 = -0.050784584
                    else:
                        var66 = 0.05339632
                else:
                    if input[5280] >= 0.10716031:
                        if input[4068] < 0.2659549:
                            var66 = 0.025640313
                        else:
                            var66 = -0.033426296
                    else:
                        if input[6785] >= 0.11370214:
                            var66 = 0.015445213
                        else:
                            var66 = -0.119113185
            else:
                if input[6611] < 0.32068118:
                    if input[6785] >= 0.094505265:
                        var66 = -0.03560193
                    else:
                        if input[8416] >= 0.3512492:
                            var66 = -0.03593452
                        else:
                            var66 = 0.11740845
                else:
                    if input[903] < 1.5411538:
                        if input[4241] >= 0.18709259:
                            var66 = -0.033572808
                        else:
                            var66 = 0.11966745
                    else:
                        if input[263] >= 0.16165602:
                            var66 = 0.08394578
                        else:
                            var66 = -0.0029159083
    if input[8466] < 0.09015203:
        if input[8361] >= 0.024262825:
            var67 = 0.031158883
        else:
            var67 = 0.12922193
    else:
        if input[7334] < 0.53020924:
            if input[3371] >= 0.21087503:
                if input[3371] < 0.27114743:
                    var67 = -0.0859683
                else:
                    var67 = 0.015938213
            else:
                if input[5280] >= 0.11687931:
                    if input[7334] < 0.28172037:
                        var67 = -0.112830155
                    else:
                        var67 = 0.029655058
                else:
                    if input[7376] < 0.26078284:
                        if input[7334] < 0.19249639:
                            var67 = 0.015199664
                        else:
                            var67 = -0.10517382
                    else:
                        if input[1997] >= 0.17189217:
                            var67 = -0.06574525
                        else:
                            var67 = 0.1110545
        else:
            if input[6326] < 0.30980593:
                if input[6328] >= 0.28239128:
                    if input[5378] < 0.2431983:
                        var67 = -0.043316063
                    else:
                        var67 = 0.026330585
                else:
                    if input[9468] >= 0.1556249:
                        var67 = -0.037795447
                    else:
                        if input[3844] >= 0.14586645:
                            var67 = 0.03947557
                        else:
                            var67 = 0.13339582
            else:
                if input[2679] < 1.4071828:
                    if input[5531] >= 0.15608986:
                        var67 = -0.053864527
                    else:
                        if input[3609] < 0.16402929:
                            var67 = -0.011654544
                        else:
                            var67 = 0.14605969
                else:
                    if input[6978] < 0.35193887:
                        var67 = 0.11601943
                    else:
                        if input[3269] < 0.20942095:
                            var67 = 0.07947181
                        else:
                            var67 = -0.0059446595
    if input[8239] < 0.07834077:
        if input[6785] >= 0.056591045:
            var68 = 0.008610248
        else:
            var68 = 0.11376228
    else:
        if input[962] < 0.51143813:
            var68 = 0.16626436
        else:
            if input[5378] < 1.001417:
                if input[3844] >= 0.15677217:
                    if input[3844] < 0.18675244:
                        var68 = -0.006548603
                    else:
                        var68 = -0.09813535
                else:
                    if input[8867] >= 0.30845657:
                        var68 = -0.06347641
                    else:
                        if input[3916] >= 0.08860388:
                            var68 = -0.005400751
                        else:
                            var68 = 0.104446016
            else:
                if input[3844] < 1.1679463:
                    if input[3844] < 0.12264583:
                        if input[4012] >= 0.04104981:
                            var68 = 0.016688831
                        else:
                            var68 = 0.114376545
                    else:
                        if input[3825] >= 0.16606335:
                            var68 = -0.04752649
                        else:
                            var68 = 0.03841161
                else:
                    if input[6785] < 0.3671209:
                        if input[5422] >= 0.19210783:
                            var68 = -0.11635216
                        else:
                            var68 = 0.021770913
                    else:
                        if input[4170] < 1.705911:
                            var68 = 0.03973605
                        else:
                            var68 = -0.012932787
    if input[3371] < 1.1404184:
        if input[5997] >= 0.10945203:
            var69 = -0.12814644
        else:
            if input[7334] >= 0.21790832:
                if input[6785] < 0.15161943:
                    var69 = 0.083546214
                else:
                    if input[7334] < 0.2795151:
                        var69 = -0.09246552
                    else:
                        var69 = 0.0043632304
            else:
                if input[7376] < 0.25533625:
                    var69 = -0.09648257
                else:
                    if input[6785] >= 0.09085733:
                        if input[3916] < 0.14740354:
                            var69 = 0.10164209
                        else:
                            var69 = -0.0020893642
                    else:
                        if input[8239] >= 0.106361635:
                            var69 = 0.0032522907
                        else:
                            var69 = 0.118594304
    else:
        if input[7972] < 0.23276408:
            if input[8466] >= 0.11098771:
                var69 = -0.04716278
            else:
                if input[9679] >= 0.1408579:
                    if input[8239] < 0.19487584:
                        var69 = -0.09174014
                    else:
                        var69 = 0.029768055
                else:
                    if input[8626] >= 0.04488177:
                        var69 = -0.009707982
                    else:
                        if input[9679] >= 0.11414267:
                            var69 = 0.027809927
                        else:
                            var69 = 0.11311213
        else:
            if input[9679] < 0.096978545:
                if input[316] >= 0.12744412:
                    if input[4175] >= 0.15671673:
                        var69 = 0.017996704
                    else:
                        var69 = -0.10692823
                else:
                    if input[9011] < 0.4044315:
                        var69 = -0.100539744
                    else:
                        if input[3916] < 0.18550347:
                            var69 = 0.0026150409
                        else:
                            var69 = 0.098687075
            else:
                if input[710] >= 0.21040188:
                    var69 = 0.10002706
                else:
                    if input[3000] < 0.2626688:
                        if input[3480] >= 0.16981728:
                            var69 = -0.07378956
                        else:
                            var69 = 0.07697477
                    else:
                        if input[8887] >= 0.17582916:
                            var69 = 0.07768945
                        else:
                            var69 = -0.0075095645
    if input[3303] < 1.2570162:
        if input[1350] < 0.27360335:
            var70 = 0.0217595
        else:
            var70 = 0.14191014
    else:
        if input[2680] < 2.00001:
            if input[9139] < 0.22183819:
                var70 = -0.004718616
            else:
                if input[8398] >= 0.16102141:
                    var70 = 0.0134703545
                else:
                    if input[5189] < 0.19023497:
                        var70 = 0.032741856
                    else:
                        if input[3269] >= 0.15996097:
                            var70 = 0.029255737
                        else:
                            var70 = 0.1458878
        else:
            if input[5736] < 0.44086555:
                if input[8239] < 0.105920464:
                    var70 = 0.009703162
                else:
                    if input[5189] < 0.19080774:
                        var70 = 0.011856538
                    else:
                        if input[5113] >= 0.17151158:
                            var70 = 0.021243608
                        else:
                            var70 = 0.11322496
            else:
                if input[4104] < 0.6177793:
                    if input[5496] >= 0.13898797:
                        if input[5496] < 0.15757152:
                            var70 = -0.08195522
                        else:
                            var70 = 0.04239167
                    else:
                        if input[5378] >= 0.16732958:
                            var70 = -0.0043018996
                        else:
                            var70 = 0.11174548
                else:
                    if input[6730] < 0.55022043:
                        if input[5898] >= 0.1443223:
                            var70 = -0.05477528
                        else:
                            var70 = 0.13770147
                    else:
                        if input[5496] < 0.36079136:
                            var70 = 0.06052487
                        else:
                            var70 = -0.0057157143
    if input[2221] < 1.1485573:
        if input[8239] >= 0.09812241:
            var71 = -0.06026227
        else:
            if input[5280] >= 0.06378568:
                if input[965] >= 0.28639302:
                    var71 = 0.062133808
                else:
                    var71 = 0.22265592
            else:
                if input[2221] < 0.25882137:
                    var71 = 0.10427778
                else:
                    if input[9679] >= 0.12205061:
                        var71 = -0.03518163
                    else:
                        var71 = 0.049618214
    else:
        if input[4666] < 0.09377654:
            if input[9679] >= 0.054987386:
                var71 = 0.03312429
            else:
                var71 = 0.1505087
        else:
            if input[9006] < 0.6959612:
                var71 = 0.113697484
            else:
                if input[1316] < 0.41825908:
                    var71 = 0.16725855
                else:
                    if input[2344] < 0.2415315:
                        if input[248] < 0.21890488:
                            var71 = 0.110126495
                        else:
                            var71 = 0.0363653
                    else:
                        if input[9525] < 1.241941:
                            var71 = 0.15685807
                        else:
                            var71 = -0.0039805896
    if input[8424] < 0.13282582:
        if input[8239] < 0.0798787:
            var72 = 0.021802839
        else:
            if input[3916] < 0.0844072:
                var72 = 0.035761535
            else:
                var72 = 0.15848094
    else:
        if input[3641] < 0.23051533:
            if input[2201] < 0.2552148:
                var72 = -0.056984622
            else:
                if input[8626] >= 0.12920874:
                    if input[3609] < 0.15816683:
                        var72 = 0.027458495
                    else:
                        var72 = -0.037172835
                else:
                    if input[8071] < 0.30003503:
                        var72 = 0.0073800706
                    else:
                        if input[3657] < 0.3054391:
                            var72 = 0.008207831
                        else:
                            var72 = 0.10343664
        else:
            if input[3609] >= 0.10058274:
                if input[6785] >= 0.07837171:
                    if input[8626] < 0.2032789:
                        var72 = -0.09515011
                    else:
                        if input[9679] < 0.22241081:
                            var72 = -0.06855871
                        else:
                            var72 = 0.08055233
                else:
                    if input[5280] < 0.12951688:
                        if input[8811] >= 0.24322988:
                            var72 = 0.1194755
                        else:
                            var72 = -0.0088210935
                    else:
                        if input[7224] >= 0.17535335:
                            var72 = 0.051579226
                        else:
                            var72 = -0.09064594
            else:
                if input[7141] >= 0.13388465:
                    if input[7152] >= 0.2717267:
                        var72 = 0.1063751
                    else:
                        if input[8466] >= 0.120436914:
                            var72 = 0.07732751
                        else:
                            var72 = -0.082734354
                else:
                    if input[5860] >= 0.14867797:
                        var72 = -0.1085295
                    else:
                        if input[2033] < 0.3023872:
                            var72 = -0.10872344
                        else:
                            var72 = 0.004078
    if input[5869] >= 0.21954967:
        var73 = -0.10986178
    else:
        if input[6306] < 0.60670716:
            var73 = -0.11161935
        else:
            if input[8923] >= 0.17807269:
                if input[4012] >= 0.1563553:
                    var73 = 0.023205215
                else:
                    if input[5113] < 0.16925263:
                        var73 = -0.0022227527
                    else:
                        if input[5587] < 0.85903096:
                            var73 = -0.007531526
                        else:
                            var73 = -0.121874206
            else:
                if input[4607] >= 0.15140015:
                    if input[4607] < 0.21681306:
                        if input[6785] >= 0.11240005:
                            var73 = 0.03984534
                        else:
                            var73 = -0.101577334
                    else:
                        if input[4607] < 0.26056585:
                            var73 = -0.0012483037
                        else:
                            var73 = 0.044795778
                else:
                    if input[4012] >= 0.12520215:
                        if input[4666] < 0.13787055:
                            var73 = -0.113621
                        else:
                            var73 = -0.028671896
                    else:
                        if input[1812] >= 0.20596889:
                            var73 = -0.11493579
                        else:
                            var73 = 0.0067845224
    if input[3916] < 0.083009586:
        if input[5025] >= 0.08001938:
            var74 = -0.0014751006
        else:
            if input[5189] < 0.212911:
                var74 = 0.020380134
            else:
                var74 = 0.12251299
    else:
        if input[892] >= 0.19968705:
            var74 = 0.2457025
        else:
            if input[5189] < 1.1187681:
                if input[4666] >= 0.12500043:
                    if input[5189] < 0.22898936:
                        if input[4666] < 0.14174043:
                            var74 = -0.06307085
                        else:
                            var74 = 0.05702103
                    else:
                        var74 = -0.091165245
                else:
                    if input[9916] >= 0.13163704:
                        var74 = -0.09082029
                    else:
                        if input[9666] < 0.54800874:
                            var74 = -0.048259344
                        else:
                            var74 = 0.08053865
            else:
                if input[2677] < 2.00001:
                    if input[8239] < 0.10301286:
                        var74 = -0.0077990415
                    else:
                        if input[5531] >= 0.14282015:
                            var74 = 0.007316302
                        else:
                            var74 = 0.12793845
                else:
                    if input[1144] < 0.21387675:
                        if input[1144] < 0.1954392:
                            var74 = 0.040497735
                        else:
                            var74 = 0.114203274
                    else:
                        if input[8626] >= 0.11628212:
                            var74 = -0.046094533
                        else:
                            var74 = -0.0014962101
    if input[5990] < 1.0:
        var75 = 0.149733
    else:
        if input[5280] < 0.048058297:
            var75 = 0.11162998
        else:
            if input[5476] < 0.25431868:
                if input[6785] >= 0.10967751:
                    if input[6878] < 0.38442856:
                        var75 = -0.08105702
                    else:
                        if input[5476] < 0.23711544:
                            var75 = -0.010881099
                        else:
                            var75 = 0.042412564
                else:
                    if input[8317] < 0.42630324:
                        var75 = -0.049635984
                    else:
                        if input[5478] >= 0.30021042:
                            var75 = -0.046933923
                        else:
                            var75 = 0.09117951
            else:
                if input[5459] < 0.48933083:
                    if input[8466] >= 0.13314956:
                        if input[8466] < 0.15118745:
                            var75 = -0.036148187
                        else:
                            var75 = 0.035532266
                    else:
                        if input[5997] >= 0.11235983:
                            var75 = 0.02759485
                        else:
                            var75 = 0.1330676
                else:
                    if input[6465] < 0.5095433:
                        if input[6785] >= 0.12599452:
                            var75 = -0.0021447286
                        else:
                            var75 = 0.13276352
                    else:
                        if input[7224] < 0.18973956:
                            var75 = 0.060805272
                        else:
                            var75 = -0.0054866746
    if input[1779] < 1.2002537:
        if input[1781] >= 0.3292537:
            var76 = 0.04467745
        else:
            var76 = 0.20063381
    else:
        if input[4068] >= 0.15810794:
            if input[4076] >= 0.27483052:
                if input[4068] < 0.21971458:
                    var76 = -0.04839915
                else:
                    var76 = 0.052350223
            else:
                if input[5280] >= 0.11019604:
                    if input[4068] < 0.26972327:
                        var76 = 0.023944752
                    else:
                        var76 = -0.024913479
                else:
                    if input[6785] >= 0.11370214:
                        var76 = 0.013251233
                    else:
                        if input[8626] >= 0.13929683:
                            var76 = 0.01234188
                        else:
                            var76 = -0.120911084
        else:
            if input[2111] < 0.0937234:
                var76 = 0.14439875
            else:
                if input[8215] < 0.27118993:
                    if input[8466] < 0.15118745:
                        var76 = 0.11259115
                    else:
                        if input[6785] < 0.22547413:
                            var76 = 0.04999743
                        else:
                            var76 = -0.0026927472
                else:
                    if input[4269] < 0.36174938:
                        if input[5280] >= 0.0737354:
                            var76 = 0.21264558
                        else:
                            var76 = -0.0049579544
                    else:
                        if input[2104] >= 0.21082489:
                            var76 = 0.11357101
                        else:
                            var76 = -0.0020862396
    if input[6418] < 0.3157804:
        if input[70] >= 0.23681445:
            var77 = -0.07705508
        else:
            if input[3214] >= 0.24440584:
                var77 = -0.052695133
            else:
                if input[7081] >= 0.16598743:
                    var77 = -0.053934004
                else:
                    if input[4921] >= 0.20661166:
                        var77 = -0.03265097
                    else:
                        if input[3217] >= 0.23695405:
                            var77 = 0.026087314
                        else:
                            var77 = 0.13580158
    else:
        if input[1350] < 0.28775734:
            if input[6785] < 0.1337869:
                if input[1350] < 0.19585028:
                    var77 = -0.016752115
                else:
                    var77 = -0.10393611
            else:
                if input[531] >= 0.13432616:
                    var77 = -0.01779338
                else:
                    if input[8239] >= 0.120521255:
                        var77 = 0.032528777
                    else:
                        if input[8466] < 0.13955419:
                            var77 = 0.033665646
                        else:
                            var77 = 0.12008039
        else:
            if input[6611] < 0.32068118:
                if input[8416] >= 0.3512492:
                    var77 = -0.030681947
                else:
                    if input[6785] >= 0.09799725:
                        var77 = -0.03885063
                    else:
                        if input[2144] < 0.61254185:
                            var77 = -0.032097407
                        else:
                            var77 = 0.11718746
            else:
                if input[7334] < 0.53020924:
                    if input[3371] >= 0.21087503:
                        if input[3371] < 0.27114743:
                            var77 = -0.07405071
                        else:
                            var77 = 0.0140637625
                    else:
                        if input[5280] >= 0.11687931:
                            var77 = -0.06711832
                        else:
                            var77 = 0.08717467
                else:
                    if input[963] < 0.7767438:
                        if input[4170] >= 0.11331123:
                            var77 = -0.033001076
                        else:
                            var77 = 0.1533932
                    else:
                        if input[9468] < 0.1372231:
                            var77 = 0.09217681
                        else:
                            var77 = -0.0057136626
    if input[710] >= 0.21040188:
        var78 = 0.09836907
    else:
        if input[8398] < 0.12470183:
            if input[284] >= 0.040838227:
                var78 = 0.005626918
            else:
                if input[5113] < 0.11082348:
                    var78 = 0.026878608
                else:
                    var78 = 0.14365865
        else:
            if input[5694] < 1.1638609:
                if input[284] < 0.26013443:
                    if input[284] < 0.1490245:
                        var78 = 0.027833546
                    else:
                        if input[5694] < 0.22920443:
                            var78 = -0.09828557
                        else:
                            var78 = -0.015459711
                else:
                    if input[4259] >= 0.28400993:
                        if input[4255] < 0.2921751:
                            var78 = -0.071115814
                        else:
                            var78 = 0.057556726
                    else:
                        if input[4255] < 0.21664993:
                            var78 = 0.08092095
                        else:
                            var78 = 0.20757167
            else:
                if input[7972] < 0.37534994:
                    if input[7445] < 0.30723515:
                        var78 = -0.11591641
                    else:
                        if input[8378] >= 0.19706315:
                            var78 = -0.0974727
                        else:
                            var78 = 0.08846646
                else:
                    if input[9006] < 0.6959612:
                        var78 = 0.11149782
                    else:
                        if input[5736] < 0.44086555:
                            var78 = 0.097144686
                        else:
                            var78 = -0.0049818335
    if input[5372] < 1.2109509:
        if input[5280] < 0.0730186:
            var79 = 0.055763956
        else:
            var79 = 0.20486382
    else:
        if input[4012] < 0.101977214:
            if input[8239] >= 0.07834077:
                var79 = 0.0043549677
            else:
                if input[3916] < 0.07751899:
                    var79 = 0.032522094
                else:
                    var79 = 0.15401326
        else:
            if input[9386] < 0.5192439:
                var79 = 0.107399344
            else:
                if input[6404] < 1.1448104:
                    var79 = 0.15375929
                else:
                    if input[3844] < 0.14459579:
                        if input[9379] >= 0.1262172:
                            var79 = -0.118428744
                        else:
                            var79 = 0.09564494
                    else:
                        if input[2762] < 0.22651482:
                            var79 = 0.1073931
                        else:
                            var79 = -0.004015691
    if input[8239] < 0.07834077:
        if input[6785] >= 0.05536623:
            var80 = 0.008370745
        else:
            var80 = 0.111615956
    else:
        if input[903] < 1.5411538:
            if input[4241] >= 0.18709259:
                var80 = -0.034120843
            else:
                if input[873] < 0.30677226:
                    var80 = -0.012141901
                else:
                    if input[6785] >= 0.08898466:
                        if input[903] < 0.30551863:
                            var80 = 0.07277312
                        else:
                            var80 = -0.011712714
                    else:
                        if input[3269] >= 0.05625376:
                            var80 = 0.025927847
                        else:
                            var80 = 0.12599643
        else:
            if input[2679] < 1.4071828:
                if input[5531] >= 0.15608986:
                    var80 = -0.050695527
                else:
                    if input[3609] < 0.16402929:
                        var80 = -0.014942102
                    else:
                        if input[5189] >= 0.18266423:
                            var80 = 0.00937599
                        else:
                            var80 = 0.13824962
            else:
                if input[8215] < 0.27118993:
                    if input[5280] < 0.099889286:
                        var80 = -0.018020643
                    else:
                        var80 = 0.111108266
                else:
                    if input[2658] < 0.38888296:
                        if input[9916] >= 0.13010333:
                            var80 = -0.07770688
                        else:
                            var80 = 0.107648015
                    else:
                        if input[2680] < 2.00001:
                            var80 = 0.117719844
                        else:
                            var80 = -0.00450905
    if input[749] < 0.13306631:
        var81 = 0.14407672
    else:
        if input[962] < 0.51143813:
            if input[8239] < 0.15410501:
                var81 = 0.047777306
            else:
                var81 = 0.15797268
        else:
            if input[8792] >= 0.13564949:
                if input[6785] >= 0.10455085:
                    var81 = 0.020178137
                else:
                    if input[2957] < 0.24613264:
                        var81 = -0.014828685
                    else:
                        var81 = -0.10615396
            else:
                if input[4836] >= 0.20206447:
                    var81 = -0.10930195
                else:
                    if input[5869] >= 0.21954967:
                        var81 = -0.107228376
                    else:
                        if input[5531] >= 0.11662267:
                            var81 = -0.042503305
                        else:
                            var81 = 0.0033395651
    if input[8466] < 0.09015203:
        if input[8239] >= 0.03928881:
            var82 = 0.033053037
        else:
            var82 = 0.12463578
    else:
        if input[4326] < 1.6494892:
            if input[4326] < 0.36764124:
                if input[4326] < 0.31650817:
                    if input[6785] >= 0.09864448:
                        var82 = -0.021681799
                    else:
                        var82 = 0.1584553
                else:
                    var82 = -0.07477166
            else:
                var82 = 0.25509107
        else:
            if input[2997] < 0.67106825:
                if input[5280] >= 0.06378568:
                    var82 = 0.17209153
                else:
                    if input[6785] < 0.16647746:
                        if input[531] >= 0.14237654:
                            var82 = 0.012568134
                        else:
                            var82 = 0.12722261
                    else:
                        if input[2997] < 0.17827815:
                            var82 = 0.09710292
                        else:
                            var82 = 0.024975844
            else:
                if input[1144] < 0.21387675:
                    if input[1144] < 0.1954392:
                        var82 = 0.039182674
                    else:
                        var82 = 0.111819245
                else:
                    if input[2104] >= 0.21082489:
                        var82 = 0.11161597
                    else:
                        if input[3878] < 0.3738104:
                            var82 = 0.100996844
                        else:
                            var82 = -0.0040151863
    if input[2033] < 0.3023872:
        if input[2033] < 0.22398409:
            var83 = -0.026230037
        else:
            var83 = -0.10828071
    else:
        if input[5378] < 0.18943173:
            if input[4104] >= 0.15589412:
                var83 = 0.018256996
            else:
                if input[9679] >= 0.09221593:
                    var83 = 0.023238512
                else:
                    if input[5386] >= 0.22302721:
                        var83 = 0.02540031
                    else:
                        var83 = 0.12531172
        else:
            if input[9525] < 1.241941:
                var83 = 0.14785025
            else:
                if input[9916] < 0.10695686:
                    if input[4666] >= 0.07481452:
                        var83 = -0.051909495
                    else:
                        if input[2562] < 0.16933328:
                            var83 = -0.0015703855
                        else:
                            var83 = 0.13610499
                else:
                    if input[9307] < 0.11202816:
                        if input[8239] >= 0.052038617:
                            var83 = 0.0357223
                        else:
                            var83 = 0.14149094
                    else:
                        if input[710] >= 0.21040188:
                            var83 = 0.09690183
                        else:
                            var83 = -0.0023672506
    if input[3371] < 1.1404184:
        if input[5997] >= 0.10945203:
            var84 = -0.114464976
        else:
            if input[7376] < 0.2658061:
                var84 = -0.09520944
            else:
                if input[7334] >= 0.21790832:
                    if input[6785] < 0.15161943:
                        var84 = 0.079203814
                    else:
                        if input[3371] < 0.26744005:
                            var84 = -0.07906035
                        else:
                            var84 = 0.009945017
                else:
                    if input[6154] < 0.2281339:
                        var84 = -0.09563835
                    else:
                        if input[3381] >= 0.28113464:
                            var84 = -0.08934405
                        else:
                            var84 = 0.090537116
    else:
        if input[5990] < 1.0:
            var84 = 0.14876789
        else:
            if input[6730] < 0.55022043:
                if input[5496] >= 0.10673534:
                    var84 = -0.07095884
                else:
                    if input[6785] >= 0.08058731:
                        if input[6730] < 0.25167024:
                            var84 = 0.006064736
                        else:
                            var84 = 0.076145
                    else:
                        if input[8398] < 0.32895666:
                            var84 = 0.036857553
                        else:
                            var84 = 0.1530697
            else:
                if input[5496] < 0.7983319:
                    if input[3844] < 0.21633394:
                        if input[3844] < 0.14586645:
                            var84 = 0.06962744
                        else:
                            var84 = -0.07589
                    else:
                        if input[8424] < 0.1858038:
                            var84 = -0.03201483
                        else:
                            var84 = 0.0650705
                else:
                    if input[3844] < 0.22016212:
                        if input[3825] >= 0.16606335:
                            var84 = -0.03604393
                        else:
                            var84 = 0.055646587
                    else:
                        if input[5476] < 0.31665754:
                            var84 = 0.07084601
                        else:
                            var84 = -0.0072175884
    if input[908] < 1.8546642:
        if input[908] < 0.24835934:
            var85 = 0.05825981
        else:
            var85 = 0.21041358
    else:
        if input[1316] < 0.41825908:
            var85 = 0.1548854
        else:
            if input[9006] < 0.6959612:
                var85 = 0.10976668
            else:
                if input[6785] < 0.06657117:
                    if input[6785] < 0.05536623:
                        if input[7224] >= 0.05478097:
                            var85 = 0.0054058586
                        else:
                            var85 = 0.1071229
                    else:
                        if input[6785] < 0.056591045:
                            var85 = -0.029357
                        else:
                            var85 = 0.062545516
                else:
                    if input[1169] >= 0.21412548:
                        if input[5280] >= 0.108336665:
                            var85 = 0.0023692865
                        else:
                            var85 = 0.11016994
                    else:
                        if input[2344] < 0.2415315:
                            var85 = 0.104189396
                        else:
                            var85 = -0.0037863466
    if input[4666] < 0.09377654:
        if input[9679] >= 0.052778974:
            var86 = 0.03694962
        else:
            var86 = 0.14633946
    else:
        if input[1779] < 1.2002537:
            if input[1781] >= 0.3292537:
                var86 = 0.04386213
            else:
                if input[1779] < 0.41562122:
                    var86 = 0.1960923
                else:
                    var86 = 0.055940457
        else:
            if input[3303] < 1.2570162:
                if input[4104] >= 0.1731243:
                    var86 = 0.019216882
                else:
                    var86 = 0.13098517
            else:
                if input[2677] < 2.00001:
                    if input[9679] >= 0.13621159:
                        var86 = -0.020268748
                    else:
                        if input[8239] < 0.10301286:
                            var86 = -0.0053935265
                        else:
                            var86 = 0.11709517
                else:
                    if input[5736] < 0.44086555:
                        if input[9468] < 0.14319532:
                            var86 = 0.0057150307
                        else:
                            var86 = 0.10013107
                    else:
                        if input[3641] < 0.2786712:
                            var86 = 0.07677547
                        else:
                            var86 = -0.0039992924
    if input[5280] < 0.048058297:
        var87 = 0.109263144
    else:
        if input[5463] < 1.0120678:
            if input[8466] >= 0.12948234:
                var87 = -0.009461659
            else:
                if input[8738] < 0.40595648:
                    var87 = 0.020814829
                else:
                    if input[3844] < 0.18173622:
                        var87 = 0.018479163
                    else:
                        var87 = 0.15583874
        else:
            if input[2221] < 1.1485573:
                if input[8239] >= 0.09812241:
                    var87 = -0.06054247
                else:
                    if input[5280] < 0.21638642:
                        if input[965] >= 0.28639302:
                            var87 = 0.051472764
                        else:
                            var87 = 0.19540466
                    else:
                        if input[2221] < 0.25882137:
                            var87 = 0.09304504
                        else:
                            var87 = 0.014930834
            else:
                if input[6418] < 0.34574562:
                    if input[70] >= 0.23681445:
                        var87 = -0.07254533
                    else:
                        if input[3214] >= 0.24440584:
                            var87 = -0.050317835
                        else:
                            var87 = 0.09842491
                else:
                    if input[1664] < 0.22088127:
                        if input[6785] >= 0.08784678:
                            var87 = -0.042912357
                        else:
                            var87 = 0.10695864
                    else:
                        if input[892] >= 0.19968705:
                            var87 = 0.21802197
                        else:
                            var87 = -0.00377548
    if input[6306] < 0.60670716:
        var88 = -0.108208634
    else:
        if input[8923] >= 0.17807269:
            if input[4012] >= 0.1563553:
                var88 = 0.022949517
            else:
                if input[6785] < 0.11651612:
                    if input[6785] < 0.111001454:
                        var88 = -0.032380443
                    else:
                        var88 = 0.0000007725674
                else:
                    if input[5587] < 0.85903096:
                        var88 = 0.013390887
                    else:
                        if input[9679] < 0.10847709:
                            var88 = -0.017001083
                        else:
                            var88 = -0.12389191
        else:
            if input[2722] >= 0.18979347:
                var88 = -0.107246
            else:
                if input[5422] >= 0.1981499:
                    if input[1347] < 0.46965322:
                        var88 = 0.03886774
                    else:
                        if input[6480] < 0.31136683:
                            var88 = -0.013614488
                        else:
                            var88 = -0.12539138
                else:
                    if input[5860] >= 0.14867797:
                        if input[9679] < 0.25961855:
                            var88 = 0.01789458
                        else:
                            var88 = -0.10312567
                    else:
                        if input[284] >= 0.1436193:
                            var88 = -0.054846525
                        else:
                            var88 = 0.00528698
    if input[8424] < 0.13282582:
        if input[8239] >= 0.06659564:
            var89 = 0.02867432
        else:
            var89 = 0.14200525
    else:
        if input[8239] < 0.08123742:
            if input[8466] >= 0.053112514:
                var89 = 0.007779119
            else:
                var89 = 0.10755151
        else:
            if input[9679] < 0.07169372:
                var89 = 0.112320855
            else:
                if input[3270] < 0.48053354:
                    var89 = 0.1080965
                else:
                    if input[2104] >= 0.21082489:
                        var89 = 0.10981306
                    else:
                        if input[710] >= 0.21040188:
                            var89 = 0.095514804
                        else:
                            var89 = -0.0038496386
    if input[2782] >= 0.16067313:
        if input[9679] < 0.11076323:
            var90 = 0.028154133
        else:
            if input[4170] >= 0.13133019:
                var90 = -0.05540734
            else:
                if input[5997] < 0.33748817:
                    var90 = -0.0377833
                else:
                    var90 = -0.12796077
    else:
        if input[8398] < 0.1410837:
            if input[5280] >= 0.064875074:
                var90 = -0.013444371
            else:
                if input[6785] < 0.0825758:
                    var90 = -0.000043299235
                else:
                    if input[9468] >= 0.09503069:
                        var90 = 0.028618572
                    else:
                        var90 = 0.1445254
        else:
            if input[4170] < 1.705911:
                if input[316] >= 0.12265691:
                    if input[9213] >= 0.12559608:
                        var90 = 0.10803711
                    else:
                        if input[7103] < 0.2999532:
                            var90 = 0.12157314
                        else:
                            var90 = -0.061433204
                else:
                    if input[360] < 0.92302686:
                        var90 = -0.094134405
                    else:
                        if input[2580] < 0.7530025:
                            var90 = -0.1000911
                        else:
                            var90 = 0.08892861
            else:
                if input[3000] < 0.2626688:
                    if input[8626] < 0.13787143:
                        if input[5280] >= 0.07997862:
                            var90 = -0.098609574
                        else:
                            var90 = 0.00033046768
                    else:
                        if input[2310] >= 0.19519673:
                            var90 = -0.09310519
                        else:
                            var90 = 0.08243974
                else:
                    if input[963] < 0.7767438:
                        if input[965] >= 0.2731175:
                            var90 = 0.042351644
                        else:
                            var90 = 0.14520916
                    else:
                        if input[6785] < 0.08898466:
                            var90 = 0.05472334
                        else:
                            var90 = -0.005586511
    var91 = var35 + var36 + var37 + var38 + var39 + var40 + var41 + var42 + var43 + var44 + var45 + var46 + var47 + var48 + var49 + var50 + var51 + var52 + var53 + var54 + var55 + var56 + var57 + var58 + var59 + var60 + var61 + var62 + var63 + var64 + var65 + var66 + var67 + var68 + var69 + var70 + var71 + var72 + var73 + var74 + var75 + var76 + var77 + var78 + var79 + var80 + var81 + var82 + var83 + var84 + var85 + var86 + var87 + var88 + var89 + var90
    if input[4104] < 0.6177793:
        if input[77] < 0.28649938:
            if input[8424] >= 0.13968639:
                var92 = -0.07709016
            else:
                var92 = 0.014442292
        else:
            if input[8239] >= 0.106361635:
                if input[5496] < 0.19161716:
                    var92 = -0.068998896
                else:
                    if input[4104] < 0.2058187:
                        var92 = -0.00018964203
                    else:
                        var92 = 0.0680658
            else:
                if input[7138] >= 0.087160066:
                    var92 = -0.039339464
                else:
                    if input[5378] >= 0.21852799:
                        if input[5378] < 0.23851988:
                            var92 = -0.062382724
                        else:
                            var92 = 0.041517038
                    else:
                        if input[5496] >= 0.22720504:
                            var92 = -0.013479494
                        else:
                            var92 = 0.115107
    else:
        if input[5459] < 0.2726077:
            if input[3844] >= 0.16743967:
                var92 = -0.01237405
            else:
                if input[6465] < 0.27096546:
                    var92 = 0.029996205
                else:
                    if input[9379] < 0.1387586:
                        var92 = 0.035091557
                    else:
                        if input[5461] < 0.33402205:
                            var92 = 0.039230984
                        else:
                            var92 = 0.14902994
        else:
            if input[6404] < 1.1448104:
                var92 = 0.14572625
            else:
                if input[2491] < 0.58572185:
                    if input[8466] >= 0.10503545:
                        var92 = 0.028256774
                    else:
                        if input[4666] >= 0.12167674:
                            var92 = 0.010084221
                        else:
                            var92 = 0.14032501
                else:
                    if input[9006] < 0.6959612:
                        var92 = 0.108048595
                    else:
                        if input[9386] < 0.22473142:
                            var92 = 0.105626516
                        else:
                            var92 = -0.0038807539
    if input[4607] >= 0.15140015:
        if input[4607] < 0.21681306:
            if input[6785] >= 0.11240005:
                var93 = 0.039748836
            else:
                if input[5280] < 0.11547779:
                    var93 = -0.015870139
                else:
                    var93 = -0.10198659
        else:
            if input[4607] < 0.26056585:
                var93 = -0.0006749213
            else:
                var93 = 0.0455967
    else:
        if input[263] < 0.23401643:
            if input[5280] >= 0.092969336:
                var93 = -0.076396205
            else:
                if input[8466] >= 0.10793656:
                    var93 = -0.06809141
                else:
                    if input[9679] >= 0.11105109:
                        var93 = -0.034020256
                    else:
                        if input[3609] < 0.15450665:
                            var93 = -0.022565402
                        else:
                            var93 = 0.093663625
        else:
            if input[9048] >= 0.17829028:
                var93 = 0.25155708
            else:
                if input[3916] < 0.083009586:
                    if input[5025] >= 0.05070388:
                        var93 = -0.0019923516
                    else:
                        if input[9916] >= 0.076358944:
                            var93 = 0.026321618
                        else:
                            var93 = 0.11917587
                else:
                    if input[3236] >= 0.22634567:
                        var93 = 0.17227203
                    else:
                        if input[1169] >= 0.21412548:
                            var93 = 0.10191742
                        else:
                            var93 = -0.002029146
    if input[9274] >= 0.21646078:
        var94 = -0.106741086
    else:
        if input[2033] < 0.3023872:
            if input[809] < 0.19113877:
                var94 = -0.10670351
            else:
                var94 = -0.02728796
        else:
            if input[5869] >= 0.21954967:
                var94 = -0.10478898
            else:
                if input[6775] >= 0.14890413:
                    if input[5676] >= 0.22542231:
                        var94 = 0.012089415
                    else:
                        if input[4012] >= 0.1101426:
                            var94 = -0.00015966712
                        else:
                            var94 = -0.12719105
                else:
                    if input[4068] >= 0.15810794:
                        if input[5280] >= 0.08238211:
                            var94 = -0.018350666
                        else:
                            var94 = -0.09518229
                    else:
                        if input[6306] < 0.60670716:
                            var94 = -0.10626464
                        else:
                            var94 = 0.00442648
    if input[5694] < 1.1638609:
        if input[284] < 0.26013443:
            if input[284] < 0.1490245:
                var95 = 0.03268052
            else:
                if input[4255] >= 0.2313131:
                    var95 = -0.011357667
                else:
                    var95 = -0.09879193
        else:
            if input[4259] >= 0.28400993:
                if input[4255] < 0.2921751:
                    var95 = -0.0671949
                else:
                    var95 = 0.051692247
            else:
                if input[4170] < 0.13808191:
                    var95 = -0.021040197
                else:
                    if input[4666] >= 0.12530945:
                        var95 = 0.010013926
                    else:
                        if input[4259] < 0.24212272:
                            var95 = 0.07183762
                        else:
                            var95 = 0.19611488
    else:
        if input[903] < 1.5411538:
            if input[4241] >= 0.18709259:
                var95 = -0.03858521
            else:
                if input[873] >= 0.278116:
                    var95 = -0.011236882
                else:
                    if input[6785] < 0.1419968:
                        if input[903] < 0.25473404:
                            var95 = 0.07152084
                        else:
                            var95 = -0.017113978
                    else:
                        if input[3371] < 0.19998325:
                            var95 = 0.029216731
                        else:
                            var95 = 0.123175636
        else:
            if input[2762] < 0.22651482:
                if input[8466] >= 0.10745005:
                    var95 = -0.028199552
                else:
                    if input[3916] >= 0.09596764:
                        var95 = 0.005509347
                    else:
                        var95 = 0.1162534
            else:
                if input[3844] < 1.1679463:
                    if input[6785] >= 0.0974956:
                        if input[3825] < 0.27538994:
                            var95 = -0.09203286
                        else:
                            var95 = 0.0006351825
                    else:
                        if input[9307] >= 0.1884855:
                            var95 = -0.066054486
                        else:
                            var95 = 0.053102132
                else:
                    if input[5378] < 1.001417:
                        if input[8867] >= 0.30845657:
                            var95 = -0.055526327
                        else:
                            var95 = 0.08524284
                    else:
                        if input[5476] < 1.171566:
                            var95 = 0.061175473
                        else:
                            var95 = -0.0064874403
    if input[5372] < 1.2109509:
        if input[5280] < 0.0730186:
            var96 = 0.051445395
        else:
            var96 = 0.18212692
    else:
        if input[2679] < 1.4071828:
            if input[5531] >= 0.15608986:
                var96 = -0.04461641
            else:
                if input[3609] < 0.16402929:
                    var96 = -0.015352665
                else:
                    if input[5189] >= 0.18180099:
                        var96 = 0.012724883
                    else:
                        if input[2679] < 0.51325023:
                            var96 = 0.13428919
                        else:
                            var96 = 0.039679293
        else:
            if input[2658] < 0.38888296:
                if input[9916] >= 0.13010333:
                    var96 = -0.072856516
                else:
                    if input[8239] >= 0.116955526:
                        var96 = -0.0043837666
                    else:
                        if input[5113] < 0.15803108:
                            var96 = 0.022421882
                        else:
                            var96 = 0.123759665
            else:
                if input[2680] < 2.00001:
                    if input[4666] < 0.14521445:
                        var96 = -0.0027129836
                    else:
                        if input[8239] >= 0.13896155:
                            var96 = 0.008403082
                        else:
                            var96 = 0.12619722
                else:
                    if input[6611] < 0.32068118:
                        if input[3825] < 0.1726117:
                            var96 = -0.054081123
                        else:
                            var96 = 0.09572103
                    else:
                        if input[2677] < 2.00001:
                            var96 = 0.09768744
                        else:
                            var96 = -0.00396217
    if input[2111] < 0.0937234:
        var97 = 0.13785456
    else:
        if input[4269] < 0.36174938:
            if input[5280] >= 0.0737354:
                var97 = 0.17941181
            else:
                var97 = -0.007912963
        else:
            if input[1779] < 1.2002537:
                if input[1781] >= 0.3527842:
                    var97 = 0.052425068
                else:
                    if input[1779] < 0.41562122:
                        var97 = 0.18114915
                    else:
                        var97 = 0.05270174
            else:
                if input[3504] < 1.4315313:
                    if input[6785] < 0.16000375:
                        if input[3504] < 0.2531703:
                            var97 = 0.027405757
                        else:
                            var97 = -0.03425587
                    else:
                        if input[8239] >= 0.10301286:
                            var97 = -0.00530515
                        else:
                            var97 = 0.16299506
                else:
                    if input[7334] < 0.53020924:
                        if input[7376] >= 0.166726:
                            var97 = -0.08915698
                        else:
                            var97 = 0.06530197
                    else:
                        if input[5189] < 1.1187681:
                            var97 = 0.056537148
                        else:
                            var97 = -0.004339423
    if input[4012] < 0.085529275:
        var98 = 0.15064
    else:
        if input[2920] < 1.1525533:
            var98 = 0.18050136
        else:
            if input[4326] < 1.6494892:
                if input[4326] < 0.36764124:
                    if input[4326] < 0.31650817:
                        if input[4327] >= 0.28864023:
                            var98 = 0.002482335
                        else:
                            var98 = 0.14204744
                    else:
                        var98 = -0.07145472
                else:
                    var98 = 0.2185463
            else:
                if input[3878] < 0.3738104:
                    if input[3880] >= 0.2621809:
                        var98 = -0.102961235
                    else:
                        if input[3844] >= 0.16662619:
                            var98 = -0.08843865
                        else:
                            var98 = 0.14866175
                else:
                    if input[2997] < 0.67106825:
                        if input[6310] >= 0.17728473:
                            var98 = 0.1846818
                        else:
                            var98 = 0.08623629
                    else:
                        if input[9525] < 1.241941:
                            var98 = 0.14142166
                        else:
                            var98 = -0.0030757382
    if input[3891] < 1.3429911:
        var99 = 0.20485277
    else:
        if input[6978] < 0.35193887:
            var99 = 0.10930593
        else:
            if input[5990] < 1.0:
                var99 = 0.13529746
            else:
                if input[3270] < 0.48053354:
                    var99 = 0.10672132
                else:
                    if input[6465] < 0.5095433:
                        if input[5459] >= 0.24023324:
                            var99 = -0.037926037
                        else:
                            var99 = 0.10760709
                    else:
                        if input[2450] >= 0.17387465:
                            var99 = -0.09284557
                        else:
                            var99 = -0.0018175278
    if input[8178] < 0.9718113:
        var100 = -0.13445507
    else:
        if input[9468] < 0.16136196:
            if input[8626] < 0.120596044:
                if input[9468] < 0.13675946:
                    var100 = -0.003946495
                else:
                    var100 = -0.09892778
            else:
                if input[8185] >= 0.17975031:
                    var100 = -0.0918188
                else:
                    if input[8239] >= 0.10109547:
                        if input[9468] < 0.15061286:
                            var100 = -0.06473043
                        else:
                            var100 = 0.113991104
                    else:
                        if input[9679] >= 0.10552166:
                            var100 = -0.051986206
                        else:
                            var100 = 0.10206467
        else:
            if input[2020] < 0.22602095:
                if input[2020] < 0.17041387:
                    if input[2020] < 0.14600074:
                        var100 = 0.078022495
                    else:
                        var100 = -0.0011651622
                else:
                    var100 = -0.09736087
            else:
                if input[6326] < 0.30980593:
                    if input[6328] >= 0.28239128:
                        var100 = -0.019883899
                    else:
                        if input[3844] < 0.1490555:
                            var100 = 0.0070867627
                        else:
                            var100 = 0.111863695
                else:
                    if input[118] < 1.1779219:
                        var100 = 0.19466837
                    else:
                        if input[2722] >= 0.18979347:
                            var100 = -0.10923641
                        else:
                            var100 = -0.00006363527
    if input[22] < 0.7892361:
        if input[3916] < 0.118260525:
            var101 = -0.016848795
        else:
            var101 = 0.14600971
    else:
        if input[5736] < 0.44086555:
            if input[9468] < 0.14319532:
                var101 = 0.0025611564
            else:
                if input[6534] >= 0.1778448:
                    var101 = 0.008904967
                else:
                    if input[2111] < 0.13591123:
                        var101 = 0.008380344
                    else:
                        if input[5113] >= 0.15968269:
                            var101 = 0.014699071
                        else:
                            var101 = 0.10863036
        else:
            if input[710] >= 0.21040188:
                var101 = 0.09428
            else:
                if input[962] < 0.51143813:
                    var101 = 0.13781518
                else:
                    if input[6321] < 0.2527963:
                        if input[9379] < 0.16865161:
                            var101 = 0.10551586
                        else:
                            var101 = 0.021571217
                    else:
                        if input[3641] < 0.23051533:
                            var101 = 0.07904563
                        else:
                            var101 = -0.0038177513
    if input[749] < 0.13306631:
        var102 = 0.1373552
    else:
        if input[9006] < 0.6959612:
            var102 = 0.10660424
        else:
            if input[4666] < 0.09377654:
                if input[9679] >= 0.049045417:
                    var102 = 0.03514469
                else:
                    var102 = 0.14279005
            else:
                if input[1316] < 0.41825908:
                    var102 = 0.14380066
                else:
                    if input[5496] < 0.7983319:
                        if input[5496] < 0.13140815:
                            var102 = 0.10732382
                        else:
                            var102 = 0.037060443
                    else:
                        if input[6730] < 0.55022043:
                            var102 = 0.112870395
                        else:
                            var102 = -0.0038980048
    if input[908] < 1.8546642:
        if input[908] < 0.37637743:
            if input[908] < 0.24835934:
                var103 = 0.052084085
            else:
                var103 = 0.21339355
        else:
            var103 = 0.06738417
    else:
        if input[5463] < 1.0120678:
            if input[8466] < 0.15199773:
                var103 = -0.005118651
            else:
                if input[8738] < 0.40595648:
                    var103 = 0.018738868
                else:
                    if input[3844] < 0.18173622:
                        var103 = 0.0152743
                    else:
                        if input[9468] >= 0.09503069:
                            var103 = 0.041284658
                        else:
                            var103 = 0.15451492
        else:
            if input[5280] < 0.048058297:
                var103 = 0.10666324
            else:
                if input[7972] < 0.37534994:
                    if input[9679] >= 0.1408579:
                        if input[7972] < 0.28900355:
                            var103 = -0.06108705
                        else:
                            var103 = 0.11068169
                    else:
                        if input[8466] >= 0.11098771:
                            var103 = -0.056719452
                        else:
                            var103 = 0.09145056
                else:
                    if input[2762] < 0.22651482:
                        if input[8466] >= 0.10745005:
                            var103 = -0.027585996
                        else:
                            var103 = 0.106885135
                    else:
                        if input[3844] < 0.13848132:
                            var103 = 0.087188825
                        else:
                            var103 = -0.003720101
    if input[4170] >= 0.21689074:
        if input[316] >= 0.17587571:
            if input[4170] < 0.27122495:
                if input[5280] < 0.1422004:
                    if input[316] < 0.26069176:
                        var104 = -0.0017509222
                    else:
                        var104 = -0.0723498
                else:
                    var104 = 0.08944309
            else:
                var104 = -0.074783705
        else:
            if input[9679] < 0.21907552:
                if input[9679] < 0.17952715:
                    var104 = -0.036193993
                else:
                    var104 = -0.0013879886
            else:
                if input[6785] < 0.18757553:
                    var104 = -0.029707028
                else:
                    if input[5280] < 0.1677061:
                        if input[4224] >= 0.3805837:
                            var104 = -0.015408973
                        else:
                            var104 = 0.15181966
                    else:
                        if input[5280] < 0.17630596:
                            var104 = -0.08630299
                        else:
                            var104 = 0.08632071
    else:
        if input[5280] >= 0.13000667:
            if input[4326] >= 0.36764124:
                var104 = 0.18756376
            else:
                if input[5372] < 1.2109509:
                    var104 = 0.16135152
                else:
                    if input[4250] < 1.5577013:
                        var104 = 0.19314405
                    else:
                        if input[5694] >= 0.37507743:
                            var104 = 0.16078477
                        else:
                            var104 = -0.048532948
        else:
            if input[4607] >= 0.15140015:
                if input[6865] < 1.1181004:
                    var104 = 0.07170579
                else:
                    if input[4607] < 0.2991694:
                        if input[5280] < 0.11547779:
                            var104 = -0.01570162
                        else:
                            var104 = -0.099299125
                    else:
                        var104 = 0.044872772
            else:
                if input[4411] >= 0.18367273:
                    if input[4411] < 0.39986807:
                        if input[1812] >= 0.1793597:
                            var104 = -0.01059842
                        else:
                            var104 = -0.13629507
                    else:
                        var104 = 0.019562816
                else:
                    if input[9274] >= 0.21646078:
                        var104 = -0.10467728
                    else:
                        if input[5869] >= 0.21954967:
                            var104 = -0.10238389
                        else:
                            var104 = 0.004400354
    if input[8923] >= 0.17807269:
        if input[4012] >= 0.1563553:
            var105 = 0.022646649
        else:
            if input[5113] < 0.16925263:
                var105 = 0.004938131
            else:
                if input[5587] < 0.85903096:
                    var105 = -0.0045898263
                else:
                    if input[6785] < 0.14139235:
                        if input[6785] < 0.114068605:
                            var105 = -0.060599137
                        else:
                            var105 = -0.01722694
                    else:
                        if input[5280] < 0.11289675:
                            var105 = -0.031898465
                        else:
                            var105 = -0.12403148
    else:
        if input[5422] >= 0.19210783:
            if input[1347] < 0.46965322:
                var105 = 0.037607692
            else:
                if input[8239] < 0.10807894:
                    var105 = -0.003921233
                else:
                    if input[8787] >= 0.22351204:
                        var105 = -0.018903729
                    else:
                        if input[1812] >= 0.23204167:
                            var105 = -0.021441069
                        else:
                            var105 = -0.12664966
        else:
            if input[531] < 0.1452476:
                if input[7334] >= 0.15945254:
                    var105 = -0.026230643
                else:
                    if input[8466] >= 0.117178135:
                        var105 = 0.0011408036
                    else:
                        if input[5280] < 0.0730186:
                            var105 = 0.022139201
                        else:
                            var105 = 0.12749073
            else:
                if input[3878] < 0.3738104:
                    if input[3880] >= 0.2621809:
                        var105 = -0.081702776
                    else:
                        if input[3844] >= 0.16662619:
                            var105 = -0.08193413
                        else:
                            var105 = 0.13342272
                else:
                    if input[710] >= 0.21040188:
                        var105 = 0.097092144
                    else:
                        if input[6785] < 0.3671209:
                            var105 = 0.017664013
                        else:
                            var105 = -0.00448269
    if input[8466] < 0.08764862:
        var106 = 0.11866949
    else:
        if input[892] >= 0.19968705:
            var106 = 0.1972368
        else:
            if input[6775] >= 0.14890413:
                if input[4012] >= 0.1101426:
                    var106 = -0.0007553124
                else:
                    var106 = -0.12146043
            else:
                if input[3000] < 1.0769:
                    if input[4170] >= 0.1178779:
                        var106 = -0.091830924
                    else:
                        if input[284] < 0.27256355:
                            var106 = -0.08020514
                        else:
                            var106 = 0.066531084
                else:
                    if input[666] < 0.1821382:
                        if input[2111] >= 0.10496774:
                            var106 = -0.107094824
                        else:
                            var106 = 0.09643309
                    else:
                        if input[4269] < 0.36174938:
                            var106 = 0.15270114
                        else:
                            var106 = -0.002565504
    if input[2782] >= 0.16067313:
        if input[9679] < 0.11667014:
            var107 = 0.012060331
        else:
            if input[4170] >= 0.13133019:
                var107 = -0.053927567
            else:
                if input[5997] < 0.33748817:
                    var107 = -0.034530457
                else:
                    var107 = -0.12237799
    else:
        if input[8398] < 0.1410837:
            if input[5280] >= 0.064875074:
                var107 = -0.013500695
            else:
                if input[6785] < 0.0825758:
                    var107 = -0.0019535858
                else:
                    if input[8421] >= 0.020461183:
                        var107 = 0.024004987
                    else:
                        if input[531] >= 0.13712215:
                            var107 = 0.041146886
                        else:
                            var107 = 0.14235762
        else:
            if input[4170] < 1.705911:
                if input[316] >= 0.12265691:
                    if input[9213] >= 0.12559608:
                        if input[4175] < 0.2475148:
                            var107 = 0.029608538
                        else:
                            var107 = 0.121088706
                    else:
                        if input[3371] >= 0.14340681:
                            var107 = 0.117604755
                        else:
                            var107 = -0.05518812
                else:
                    if input[360] < 0.92302686:
                        var107 = -0.09001905
                    else:
                        if input[2580] < 0.7530025:
                            var107 = -0.09477665
                        else:
                            var107 = 0.07899626
            else:
                if input[963] < 0.7767438:
                    if input[8466] < 0.25976613:
                        var107 = 0.034819268
                    else:
                        if input[2221] < 0.25592467:
                            var107 = 0.039489213
                        else:
                            var107 = 0.14284232
                else:
                    if input[8424] < 0.1365191:
                        if input[9679] < 0.090631:
                            var107 = 0.030017344
                        else:
                            var107 = 0.13489176
                    else:
                        if input[3000] < 1.0769:
                            var107 = 0.05387448
                        else:
                            var107 = -0.0043470575
    if input[6404] < 1.1448104:
        var108 = 0.1385883
    else:
        if input[5459] < 0.2726077:
            if input[3844] >= 0.16743967:
                var108 = -0.012573591
            else:
                if input[5496] < 0.16571596:
                    var108 = 0.014263933
                else:
                    if input[9679] >= 0.11706728:
                        var108 = 0.050647944
                    else:
                        var108 = 0.13748229
        else:
            if input[2104] >= 0.21082489:
                var108 = 0.10698495
            else:
                if input[3270] < 0.48053354:
                    var108 = 0.10532843
                else:
                    if input[4952] < 0.85737866:
                        var108 = 0.19266792
                    else:
                        if input[2221] < 1.1485573:
                            var108 = 0.1090464
                        else:
                            var108 = -0.002972802
    if input[9679] < 0.07169372:
        if input[8239] >= 0.03928881:
            var109 = 0.0049766107
        else:
            var109 = 0.109408855
    else:
        if input[8239] < 0.07834077:
            var109 = 0.10795734
        else:
            if input[9386] < 0.22473142:
                var109 = 0.10319417
            else:
                if input[9006] < 0.6959612:
                    var109 = 0.105132
                else:
                    if input[3303] < 1.2570162:
                        if input[6418] >= 0.2107655:
                            var109 = 0.027976876
                        else:
                            var109 = 0.11998049
                    else:
                        if input[2387] < 0.6178757:
                            var109 = 0.1425422
                        else:
                            var109 = -0.0031915808
    if input[9048] >= 0.17829028:
        var110 = 0.22473633
    else:
        if input[2679] < 1.4071828:
            if input[5531] >= 0.15608986:
                var110 = -0.04079915
            else:
                if input[3609] < 0.16402929:
                    var110 = -0.015250406
                else:
                    if input[5189] >= 0.18180099:
                        var110 = 0.009690325
                    else:
                        if input[2679] < 0.51325023:
                            var110 = 0.12637538
                        else:
                            var110 = 0.039332584
        else:
            if input[8148] >= 0.39003524:
                var110 = 0.30214116
            else:
                if input[5378] < 0.18943173:
                    if input[8239] < 0.17391537:
                        if input[8239] < 0.09074553:
                            var110 = 0.06598439
                        else:
                            var110 = -0.0064677657
                    else:
                        if input[4555] < 0.29347193:
                            var110 = 0.02127759
                        else:
                            var110 = 0.11974479
                else:
                    if input[1169] >= 0.21412548:
                        if input[5280] >= 0.108336665:
                            var110 = -0.0028902972
                        else:
                            var110 = 0.1055247
                    else:
                        if input[809] >= 0.1527935:
                            var110 = -0.08870038
                        else:
                            var110 = -0.0013351095
    if input[8792] >= 0.13564949:
        if input[6785] >= 0.10455085:
            var111 = 0.02172826
        else:
            if input[8792] < 0.19012164:
                var111 = -0.10203057
            else:
                var111 = -0.03539096
    else:
        if input[3256] >= 0.17889965:
            if input[2910] < 0.2501131:
                var111 = -0.025570411
            else:
                var111 = -0.12723063
        else:
            if input[2658] < 0.38888296:
                if input[9916] >= 0.13010333:
                    var111 = -0.069173746
                else:
                    if input[8466] >= 0.13564149:
                        var111 = -0.019642938
                    else:
                        if input[8626] >= 0.11921643:
                            var111 = -0.015653167
                        else:
                            var111 = 0.11728126
            else:
                if input[4068] >= 0.13236088:
                    if input[5280] < 0.1561737:
                        if input[5531] < 0.15238667:
                            var111 = -0.071001776
                        else:
                            var111 = 0.011216964
                    else:
                        if input[4078] < 0.28989235:
                            var111 = 0.011281904
                        else:
                            var111 = -0.10630649
                else:
                    if input[2680] < 2.00001:
                        if input[4666] < 0.14521445:
                            var111 = -0.0036114068
                        else:
                            var111 = 0.11177235
                    else:
                        if input[5860] >= 0.14867797:
                            var111 = -0.09880059
                        else:
                            var111 = 0.001538124
    if input[911] >= 0.2729628:
        var112 = 0.23026553
    else:
        if input[4908] < 2.00001:
            var112 = -0.13419382
        else:
            if input[1812] >= 0.20596889:
                if input[4012] < 0.17289436:
                    var112 = 0.1172052
                else:
                    if input[1381] >= 0.29470038:
                        var112 = 0.024128059
                    else:
                        if input[2691] >= 0.30450696:
                            var112 = -0.0054039336
                        else:
                            var112 = -0.11480804
            else:
                if input[1321] < 0.47067758:
                    if input[6785] < 0.127921:
                        if input[1321] < 0.25102943:
                            var112 = -0.04527914
                        else:
                            var112 = 0.019061869
                    else:
                        if input[3825] >= 0.23903799:
                            var112 = -0.0216952
                        else:
                            var112 = 0.12994878
                else:
                    if input[2408] >= 0.19339931:
                        if input[5280] < 0.11656167:
                            var112 = 0.043395396
                        else:
                            var112 = -0.083289646
                    else:
                        if input[2491] < 0.48270562:
                            var112 = 0.11769573
                        else:
                            var112 = 0.0007431823
    if input[3916] < 0.083009586:
        if input[5025] < 0.28060216:
            var113 = 0.0012833675
        else:
            if input[9916] >= 0.076358944:
                var113 = 0.021719234
            else:
                var113 = 0.11616174
    else:
        if input[9916] < 0.10695686:
            if input[4666] >= 0.07269929:
                var113 = -0.043749202
            else:
                if input[5280] >= 0.064875074:
                    var113 = 0.010733661
                else:
                    if input[8239] < 0.0842448:
                        var113 = 0.027401468
                    else:
                        var113 = 0.14299248
        else:
            if input[9307] < 0.11202816:
                if input[8239] >= 0.045043744:
                    var113 = 0.03058202
                else:
                    var113 = 0.13468038
            else:
                if input[2371] < 1.2550216:
                    if input[9468] < 0.18279625:
                        var113 = 0.037556853
                    else:
                        var113 = 0.157531
                else:
                    if input[5694] < 1.1638609:
                        if input[284] < 0.26013443:
                            var113 = -0.04493102
                        else:
                            var113 = 0.10891563
                    else:
                        if input[5736] < 0.44086555:
                            var113 = 0.088601865
                        else:
                            var113 = -0.0032558867
    if input[6306] < 0.60670716:
        var114 = -0.10386522
    else:
        if input[7930] < 1.4311116:
            var114 = -0.13940671
        else:
            if input[2722] >= 0.18979347:
                var114 = -0.101522505
            else:
                if input[3891] < 1.3429911:
                    var114 = 0.18841648
                else:
                    if input[4607] < 0.26056585:
                        if input[6785] < 0.15161943:
                            var114 = 0.069198705
                        else:
                            var114 = -0.08860763
                    else:
                        if input[263] < 0.23401643:
                            var114 = 0.06861996
                        else:
                            var114 = 0.0012613994
    if input[9525] < 1.241941:
        var115 = 0.13492683
    else:
        if input[5476] < 0.20807692:
            if input[8239] < 0.11622336:
                if input[9679] >= 0.097336516:
                    var115 = -0.11596119
                else:
                    if input[5476] < 0.17586254:
                        var115 = 0.07120228
                    else:
                        var115 = 0.0032370703
            else:
                if input[3844] >= 0.14459579:
                    var115 = -0.021104531
                else:
                    if input[1350] < 0.18537076:
                        var115 = -0.026213324
                    else:
                        if input[5531] < 0.15608986:
                            var115 = -0.013916899
                        else:
                            var115 = 0.11401927
        else:
            if input[6611] < 0.29233888:
                if input[8416] < 0.42093498:
                    var115 = -0.037520986
                else:
                    if input[3825] < 0.1726117:
                        var115 = -0.03640566
                    else:
                        if input[2144] < 0.61254185:
                            var115 = -0.038729284
                        else:
                            var115 = 0.115352415
            else:
                if input[903] < 1.5411538:
                    if input[4241] >= 0.18709259:
                        var115 = -0.040006142
                    else:
                        if input[873] < 0.30677226:
                            var115 = -0.008654292
                        else:
                            var115 = 0.10082275
                else:
                    if input[1777] < 0.48005053:
                        if input[2876] < 0.3147277:
                            var115 = 0.04993839
                        else:
                            var115 = 0.19781876
                    else:
                        if input[5496] < 0.7983319:
                            var115 = 0.045774795
                        else:
                            var115 = -0.004126287
    if input[3371] < 1.1404184:
        if input[7334] >= 0.20345515:
            if input[5280] >= 0.09922387:
                var116 = 0.09412428
            else:
                if input[6785] < 0.15161943:
                    var116 = 0.040127005
                else:
                    if input[3371] < 0.26927283:
                        var116 = -0.10810053
                    else:
                        var116 = -0.008574239
        else:
            if input[3381] >= 0.28113464:
                var116 = -0.099015504
            else:
                if input[8239] < 0.25879017:
                    if input[3371] < 0.16423859:
                        var116 = 0.08458574
                    else:
                        if input[7334] >= 0.17575742:
                            var116 = 0.07292049
                        else:
                            var116 = -0.07181995
                else:
                    if input[6785] >= 0.09920907:
                        if input[3371] < 0.28921336:
                            var116 = -0.02238349
                        else:
                            var116 = 0.11165279
                    else:
                        if input[3087] >= 0.17597196:
                            var116 = -0.084465876
                        else:
                            var116 = 0.10289575
    else:
        if input[5990] < 1.0:
            var116 = 0.13574925
        else:
            if input[6978] < 0.35193887:
                var116 = 0.10693528
            else:
                if input[8626] >= 0.11628212:
                    if input[2628] < 0.40155745:
                        var116 = 0.21346532
                    else:
                        if input[6785] >= 0.10568602:
                            var116 = 0.009771741
                        else:
                            var116 = -0.0511611
                else:
                    if input[3641] < 0.2786712:
                        if input[2201] < 0.2552148:
                            var116 = -0.06855376
                        else:
                            var116 = 0.08951753
                    else:
                        if input[3609] >= 0.10058274:
                            var116 = -0.04540392
                        else:
                            var116 = 0.0007465135
    if input[9162] < 0.24447975:
        if input[9557] >= 0.16911274:
            var117 = 0.024006415
        else:
            if input[9162] < 0.21937142:
                if input[6611] >= 0.11847811:
                    var117 = 0.034212936
                else:
                    var117 = 0.14149606
            else:
                if input[9162] < 0.23261124:
                    var117 = -0.027278854
                else:
                    var117 = 0.09742515
    else:
        if input[6785] < 0.05536623:
            if input[7224] >= 0.045544073:
                var117 = 0.005725692
            else:
                var117 = 0.104323104
        else:
            if input[3270] < 0.48053354:
                var117 = 0.10385324
            else:
                if input[1779] < 1.2002537:
                    if input[1781] < 0.52772534:
                        if input[1781] < 0.3292537:
                            var117 = 0.10128498
                        else:
                            var117 = -0.041849256
                    else:
                        var117 = 0.16694473
                else:
                    if input[2677] < 0.5621878:
                        if input[5189] >= 0.17032836:
                            var117 = -0.016742783
                        else:
                            var117 = 0.10199996
                    else:
                        if input[962] < 0.51143813:
                            var117 = 0.12770231
                        else:
                            var117 = -0.0030061116
    if input[5372] < 1.2109509:
        if input[5280] < 0.09770191:
            var118 = 0.075179085
        else:
            var118 = 0.17031549
    else:
        if input[6418] < 0.3157804:
            if input[7081] >= 0.16598743:
                var118 = -0.07841165
            else:
                if input[3214] >= 0.24440584:
                    var118 = -0.053989377
                else:
                    if input[9916] >= 0.14833075:
                        var118 = -0.048965927
                    else:
                        if input[3844] < 0.15884328:
                            var118 = -0.01766789
                        else:
                            var118 = 0.109490134
        else:
            if input[4104] < 0.2589911:
                if input[77] < 0.28649938:
                    var118 = -0.052979447
                else:
                    if input[8239] >= 0.106361635:
                        if input[5496] < 0.19161716:
                            var118 = -0.069363944
                        else:
                            var118 = 0.050801393
                    else:
                        if input[5378] >= 0.21852799:
                            var118 = -0.015642492
                        else:
                            var118 = 0.1131195
            else:
                if input[6730] < 0.55022043:
                    if input[5898] >= 0.1443223:
                        var118 = -0.05458023
                    else:
                        if input[8398] >= 0.1346456:
                            var118 = -0.00849056
                        else:
                            var118 = 0.11945335
                else:
                    if input[6611] < 0.29233888:
                        if input[5997] >= 0.09214852:
                            var118 = -0.018740902
                        else:
                            var118 = 0.098684154
                    else:
                        if input[2997] < 0.67106825:
                            var118 = 0.09474237
                        else:
                            var118 = -0.003672432
    if input[8178] < 0.9718113:
        var119 = -0.12599409
    else:
        if input[9468] < 0.23882487:
            if input[5018] >= 0.20552404:
                var119 = -0.13045391
            else:
                if input[8239] >= 0.10109547:
                    if input[6785] >= 0.095677584:
                        var119 = 0.09257438
                    else:
                        if input[9679] < 0.11706728:
                            var119 = 0.05773576
                        else:
                            var119 = -0.047797035
                else:
                    if input[284] >= 0.13174155:
                        var119 = -0.09315839
                    else:
                        if input[9679] >= 0.11276664:
                            var119 = -0.031633373
                        else:
                            var119 = 0.06970199
        else:
            if input[9460] < 0.8755927:
                if input[8239] >= 0.10109547:
                    if input[6785] < 0.11077283:
                        var119 = -0.07644459
                    else:
                        if input[8239] < 0.11720029:
                            var119 = -0.043628946
                        else:
                            var119 = 0.04209172
                else:
                    if input[9679] >= 0.18978526:
                        var119 = -0.04631351
                    else:
                        if input[9535] < 0.4488435:
                            var119 = -0.045172673
                        else:
                            var119 = 0.112278335
            else:
                if input[6321] < 0.2527963:
                    var119 = 0.10047823
                else:
                    if input[2762] < 0.22651482:
                        if input[3916] >= 0.09596764:
                            var119 = -0.018512268
                        else:
                            var119 = 0.10266938
                    else:
                        if input[710] < 0.67982006:
                            var119 = 0.08944633
                        else:
                            var119 = -0.0036014852
    if input[2920] < 1.1525533:
        if input[2920] < 0.4299407:
            var120 = 0.17252508
        else:
            var120 = 0.051119883
    else:
        if input[3236] < 1.3728962:
            if input[5025] < 0.16575636:
                var120 = -0.07578503
            else:
                if input[3236] < 0.47455654:
                    var120 = 0.16993289
                else:
                    var120 = 0.048014026
        else:
            if input[2111] < 0.0937234:
                var120 = 0.13213593
            else:
                if input[6310] >= 0.2118919:
                    if input[6312] < 0.3387697:
                        if input[3000] < 0.18728551:
                            var120 = 0.059367526
                        else:
                            var120 = -0.040855233
                    else:
                        var120 = 0.18238227
                else:
                    if input[908] >= 0.24311502:
                        if input[908] < 0.46076518:
                            var120 = 0.19385472
                        else:
                            var120 = 0.05535127
                    else:
                        if input[9006] < 0.6959612:
                            var120 = 0.103760086
                        else:
                            var120 = -0.0024037145
    if input[118] < 1.1779219:
        var121 = 0.17441788
    else:
        if input[8626] < 0.0879145:
            if input[9679] < 0.07930769:
                var121 = 0.025951965
            else:
                var121 = 0.13292225
        else:
            if input[5463] < 1.0120678:
                if input[8466] < 0.15199773:
                    var121 = -0.0050020455
                else:
                    if input[8738] < 0.40595648:
                        var121 = 0.016856153
                    else:
                        if input[3844] < 0.18173622:
                            var121 = 0.014975369
                        else:
                            var121 = 0.13983026
            else:
                if input[1169] >= 0.21412548:
                    if input[5280] >= 0.108336665:
                        var121 = -0.005005594
                    else:
                        var121 = 0.10375797
                else:
                    if input[6404] < 1.1448104:
                        var121 = 0.13236575
                    else:
                        if input[7334] < 0.53020924:
                            var121 = 0.04952159
                        else:
                            var121 = -0.0032246558
    if input[4666] < 0.08900275:
        var122 = 0.13585447
    else:
        if input[9991] < 0.3367947:
            if input[9916] >= 0.13737911:
                var122 = -0.005607888
            else:
                if input[8239] >= 0.05912797:
                    var122 = 0.059399515
                else:
                    var122 = 0.17044188
        else:
            if input[1316] < 0.41825908:
                var122 = 0.13284636
            else:
                if input[2782] >= 0.16067313:
                    if input[9679] < 0.11667014:
                        var122 = 0.012890369
                    else:
                        if input[5997] < 0.33748817:
                            var122 = -0.031731542
                        else:
                            var122 = -0.11349962
                else:
                    if input[4170] < 1.705911:
                        if input[316] >= 0.11482531:
                            var122 = -0.0383698
                        else:
                            var122 = 0.06519239
                    else:
                        if input[8398] < 0.14455892:
                            var122 = 0.1008329
                        else:
                            var122 = -0.003054596
    if input[5869] >= 0.21954967:
        var123 = -0.10023304
    else:
        if input[8923] >= 0.17807269:
            if input[4012] >= 0.1563553:
                var123 = 0.022208344
            else:
                if input[6785] < 0.11651612:
                    if input[6785] < 0.111001454:
                        var123 = -0.02711825
                    else:
                        var123 = 0.0076876404
                else:
                    if input[5587] < 0.85903096:
                        var123 = 0.010896787
                    else:
                        if input[9679] < 0.11517945:
                            var123 = -0.02109149
                        else:
                            var123 = -0.113341294
        else:
            if input[3504] < 1.4315313:
                if input[6785] < 0.16000375:
                    if input[5280] < 0.10509737:
                        var123 = 0.023781847
                    else:
                        var123 = -0.034257006
                else:
                    if input[8239] >= 0.10301286:
                        var123 = -0.010356004
                    else:
                        if input[5280] >= 0.17018904:
                            var123 = 0.028621186
                        else:
                            var123 = 0.15787403
            else:
                if input[710] >= 0.21040188:
                    var123 = 0.09496229
                else:
                    if input[22] < 0.7892361:
                        if input[3916] < 0.1221415:
                            var123 = 0.0021256392
                        else:
                            var123 = 0.13775943
                    else:
                        if input[8215] < 0.27118993:
                            var123 = 0.09741985
                        else:
                            var123 = -0.0007645944
    if input[4012] < 0.085529275:
        var124 = 0.14274232
    else:
        if input[892] >= 0.19968705:
            var124 = 0.18104427
        else:
            if input[6775] >= 0.14890413:
                if input[4012] < 0.18415895:
                    var124 = 0.0007050441
                else:
                    if input[3000] >= 0.13804758:
                        var124 = -0.031233195
                    else:
                        var124 = -0.12018371
            else:
                if input[881] < 1.0:
                    var124 = 0.13035905
                else:
                    if input[2104] >= 0.21082489:
                        var124 = 0.10454046
                    else:
                        if input[5248] < 0.4452728:
                            var124 = 0.13113327
                        else:
                            var124 = -0.0013129663
    if input[4411] >= 0.18367273:
        if input[4411] < 0.39986807:
            if input[1812] >= 0.1793597:
                var125 = -0.0061903796
            else:
                if input[9679] < 0.15008332:
                    var125 = -0.032749083
                else:
                    if input[8239] < 0.14522362:
                        var125 = -0.042614818
                    else:
                        var125 = -0.13964383
        else:
            var125 = 0.02134001
    else:
        if input[2762] < 0.22651482:
            if input[3916] >= 0.09596764:
                var125 = -0.016714294
            else:
                if input[8466] >= 0.13564149:
                    var125 = -0.012281365
                else:
                    var125 = 0.112520136
        else:
            if input[6465] < 0.5095433:
                if input[5459] < 0.3057792:
                    var125 = -0.041888606
                else:
                    if input[6785] >= 0.12599452:
                        if input[6465] < 0.32386243:
                            var125 = -0.02838268
                        else:
                            var125 = 0.03211459
                    else:
                        if input[9916] < 0.13619913:
                            var125 = -0.01807959
                        else:
                            var125 = 0.12230509
            else:
                if input[5459] < 0.48933083:
                    if input[5496] < 0.18194908:
                        var125 = 0.008713777
                    else:
                        if input[5459] < 0.2726077:
                            var125 = 0.1274554
                        else:
                            var125 = 0.046050902
                else:
                    if input[5280] < 0.048058297:
                        var125 = 0.10403303
                    else:
                        if input[6978] < 0.35193887:
                            var125 = 0.1055126
                        else:
                            var125 = -0.0021741139
    if input[749] < 0.13306631:
        var126 = 0.13130824
    else:
        if input[9525] < 1.241941:
            var126 = 0.13054053
        else:
            if input[9048] < 1.1998109:
                var126 = 0.19890365
            else:
                if input[3844] < 1.1679463:
                    if input[3844] < 0.12264583:
                        if input[8239] >= 0.08879603:
                            var126 = 0.016242981
                        else:
                            var126 = 0.10451364
                    else:
                        if input[7081] >= 0.21459563:
                            var126 = -0.12377882
                        else:
                            var126 = 0.026451224
                else:
                    if input[4952] < 0.85737866:
                        var126 = 0.19323157
                    else:
                        if input[5378] < 1.001417:
                            var126 = 0.071524434
                        else:
                            var126 = -0.0038461871
    if input[5422] >= 0.1981499:
        if input[1347] < 0.46965322:
            var127 = 0.03697577
        else:
            if input[749] < 0.21985821:
                var127 = 0.0014273557
            else:
                if input[8787] >= 0.22351204:
                    var127 = -0.016720874
                else:
                    if input[9679] < 0.15008332:
                        var127 = -0.017259782
                    else:
                        var127 = -0.12341865
    else:
        if input[3878] < 0.3738104:
            if input[3880] >= 0.2621809:
                var127 = -0.07483801
            else:
                if input[3844] >= 0.16662619:
                    var127 = -0.07500736
                else:
                    if input[625] < 0.28289574:
                        var127 = -0.052218504
                    else:
                        if input[8466] >= 0.14054188:
                            var127 = -0.020153409
                        else:
                            var127 = 0.15022914
        else:
            if input[284] >= 0.1436193:
                if input[3311] < 0.19143133:
                    var127 = 0.09719632
                else:
                    if input[9679] >= 0.11382445:
                        if input[5280] >= 0.092969336:
                            var127 = 0.14742857
                        else:
                            var127 = -0.050931912
                    else:
                        if input[9284] >= 0.18384315:
                            var127 = 0.07198867
                        else:
                            var127 = -0.061618723
            else:
                if input[4170] < 1.705911:
                    if input[4175] >= 0.13300855:
                        if input[4170] < 0.12431107:
                            var127 = -0.11187786
                        else:
                            var127 = -0.012986824
                    else:
                        if input[5050] < 1.1461158:
                            var127 = -0.08278415
                        else:
                            var127 = 0.06157983
                else:
                    if input[6785] < 0.3671209:
                        if input[5280] < 0.20319402:
                            var127 = -0.027138922
                        else:
                            var127 = 0.02728347
                    else:
                        if input[3844] < 0.49723622:
                            var127 = 0.042208187
                        else:
                            var127 = -0.0075282427
    if input[8424] < 0.13282582:
        if input[8239] >= 0.06659564:
            var128 = 0.023195926
        else:
            var128 = 0.13027798
    else:
        if input[5694] < 1.1638609:
            if input[284] < 0.26013443:
                if input[284] < 0.1490245:
                    var128 = 0.038074706
                else:
                    if input[5280] >= 0.09344549:
                        var128 = -0.009946549
                    else:
                        var128 = -0.088237636
            else:
                if input[4259] >= 0.28400993:
                    if input[4255] < 0.2921751:
                        var128 = -0.06547806
                    else:
                        var128 = 0.04065814
                else:
                    if input[4170] < 0.13808191:
                        var128 = -0.023239171
                    else:
                        if input[4666] >= 0.12530945:
                            var128 = 0.00395411
                        else:
                            var128 = 0.13466454
        else:
            if input[911] < 1.1692451:
                if input[5280] < 0.07997862:
                    var128 = 0.041443788
                else:
                    var128 = 0.2053911
            else:
                if input[5531] >= 0.11662267:
                    if input[6785] >= 0.10619637:
                        if input[5280] >= 0.10129063:
                            var128 = -0.09823876
                        else:
                            var128 = 0.0696655
                    else:
                        if input[4365] >= 0.16270001:
                            var128 = 0.117196836
                        else:
                            var128 = -0.052125495
                else:
                    if input[2679] < 1.4071828:
                        if input[3609] < 0.16402929:
                            var128 = -0.013515783
                        else:
                            var128 = 0.10527184
                    else:
                        if input[263] >= 0.16165602:
                            var128 = 0.068932354
                        else:
                            var128 = -0.0011234554
    if input[8148] < 1.5330952:
        if input[8148] < 0.39003524:
            var129 = 0.06631331
        else:
            var129 = 0.26382774
    else:
        if input[8763] >= 0.17155485:
            if input[3844] >= 0.17259286:
                var129 = 0.090588
            else:
                if input[5496] >= 0.18952721:
                    var129 = 0.08155244
                else:
                    if input[7334] < 0.26582587:
                        var129 = 0.06429748
                    else:
                        if input[5280] < 0.113409996:
                            var129 = -0.015283822
                        else:
                            var129 = -0.0880298
        else:
            if input[5105] >= 0.1816007:
                var129 = -0.11211661
            else:
                if input[8792] >= 0.13564949:
                    if input[6785] >= 0.10455085:
                        var129 = 0.02226669
                    else:
                        if input[8792] < 0.19012164:
                            var129 = -0.100502394
                        else:
                            var129 = -0.033499625
                else:
                    if input[5860] >= 0.14867797:
                        if input[9679] < 0.25961855:
                            var129 = 0.01223671
                        else:
                            var129 = -0.09484185
                    else:
                        if input[2722] >= 0.18979347:
                            var129 = -0.098560356
                        else:
                            var129 = 0.0031129834
    if input[5531] < 0.10104962:
        if input[5280] < 0.052636597:
            var130 = 0.021986118
        else:
            var130 = 0.14530115
    else:
        if input[9386] < 0.5192439:
            var130 = 0.09967439
        else:
            if input[9006] < 0.6959612:
                var130 = 0.10253256
            else:
                if input[5997] >= 0.17846912:
                    if input[9379] >= 0.22294031:
                        var130 = 0.014251172
                    else:
                        if input[6785] >= 0.15759434:
                            var130 = -0.015432107
                        else:
                            var130 = -0.13083968
                else:
                    if input[5736] >= 0.18245144:
                        if input[3000] >= 0.17934027:
                            var130 = 0.009548287
                        else:
                            var130 = 0.097757675
                    else:
                        if input[9557] < 0.20949642:
                            var130 = 0.11739192
                        else:
                            var130 = -0.001760208
    if input[4326] < 1.6494892:
        if input[6785] >= 0.10143764:
            var131 = 0.006716224
        else:
            if input[9307] >= 0.13349771:
                var131 = 0.022821296
            else:
                if input[3000] >= 0.16518599:
                    var131 = 0.03446242
                else:
                    var131 = 0.14222798
    else:
        if input[2920] < 1.1525533:
            if input[2920] < 0.4299407:
                var131 = 0.16568318
            else:
                var131 = 0.04815776
        else:
            if input[2680] < 2.00001:
                if input[2680] < 0.25219274:
                    if input[2680] < 0.23268394:
                        if input[8466] < 0.11625491:
                            var131 = 0.023441536
                        else:
                            var131 = 0.09802661
                    else:
                        if input[6785] < 0.105886854:
                            var131 = -0.012979278
                        else:
                            var131 = -0.059822436
                else:
                    if input[8239] >= 0.12450814:
                        var131 = 0.02869296
                    else:
                        var131 = 0.1378555
            else:
                if input[2658] < 0.38888296:
                    if input[9916] >= 0.13010333:
                        var131 = -0.06605978
                    else:
                        if input[5113] < 0.15803108:
                            var131 = 0.010576531
                        else:
                            var131 = 0.10688895
                else:
                    if input[5476] < 1.171566:
                        if input[5494] < 1.0953032:
                            var131 = -0.09251266
                        else:
                            var131 = 0.052873712
                    else:
                        if input[5469] < 0.3876029:
                            var131 = 0.12826066
                        else:
                            var131 = -0.0034000077
    if input[4908] < 2.00001:
        var132 = -0.12676233
    else:
        if input[8218] < 1.0324993:
            if input[3916] < 0.13526714:
                var132 = 0.034667358
            else:
                if input[6785] < 0.19180253:
                    var132 = 0.04115918
                else:
                    var132 = 0.16139267
        else:
            if input[2221] < 1.1485573:
                if input[8239] >= 0.09812241:
                    var132 = -0.062190633
                else:
                    if input[5280] >= 0.06378568:
                        if input[6252] >= 0.18988906:
                            var132 = 0.04370478
                        else:
                            var132 = 0.17156474
                    else:
                        if input[6785] < 0.16795206:
                            var132 = 0.096839555
                        else:
                            var132 = 0.022007383
            else:
                if input[8239] < 0.07834077:
                    if input[6785] >= 0.025742454:
                        var132 = 0.0052890624
                    else:
                        var132 = 0.10554252
                else:
                    if input[1169] >= 0.21412548:
                        if input[5280] >= 0.108336665:
                            var132 = -0.0024908765
                        else:
                            var132 = 0.10174443
                    else:
                        if input[809] >= 0.1251306:
                            var132 = -0.07811221
                        else:
                            var132 = -0.0006301213
    if input[3256] >= 0.17889965:
        if input[2910] < 0.2501131:
            var133 = -0.021014545
        else:
            var133 = -0.11978282
    else:
        if input[2932] < 1.0769625:
            var133 = 0.15152793
        else:
            if input[3891] < 1.3429911:
                var133 = 0.17225896
            else:
                if input[1812] >= 0.20596889:
                    if input[4012] < 0.17289436:
                        var133 = 0.10661184
                    else:
                        if input[1381] < 0.38422158:
                            var133 = 0.027021408
                        else:
                            var133 = -0.10532104
                else:
                    if input[5725] < 0.22910857:
                        if input[9916] >= 0.11777075:
                            var133 = -0.048554387
                        else:
                            var133 = 0.11933203
                    else:
                        if input[7972] < 0.37534994:
                            var133 = 0.066496216
                        else:
                            var133 = -0.00069937255
    if input[1779] < 1.2002537:
        if input[1781] >= 0.3292537:
            var134 = 0.02572493
        else:
            var134 = 0.1505224
    else:
        if input[4269] < 0.29803684:
            if input[5280] >= 0.0737354:
                var134 = 0.16146956
            else:
                var134 = 0.007816316
        else:
            if input[4068] >= 0.15810794:
                if input[4078] < 0.28989235:
                    if input[4068] < 0.20591357:
                        var134 = -0.042661935
                    else:
                        var134 = 0.06372158
                else:
                    if input[5280] >= 0.10080631:
                        if input[4068] < 0.25262994:
                            var134 = 0.0060776575
                        else:
                            var134 = -0.030217392
                    else:
                        if input[6785] >= 0.11370214:
                            var134 = 0.016416976
                        else:
                            var134 = -0.100701645
            else:
                if input[908] < 1.8546642:
                    if input[908] < 0.37637743:
                        var134 = 0.17158726
                    else:
                        var134 = 0.05798594
                else:
                    if input[2387] < 0.6178757:
                        if input[9679] >= 0.10631895:
                            var134 = 0.00059592427
                        else:
                            var134 = 0.13937302
                    else:
                        if input[2677] < 0.5621878:
                            var134 = 0.08977597
                        else:
                            var134 = -0.0012721608
    if input[3889] < 1.2113615:
        var135 = 0.20740807
    else:
        if input[5819] < 0.122899115:
            if input[5280] >= 0.0066040074:
                var135 = 0.023488855
            else:
                var135 = 0.13535716
        else:
            if input[5372] < 1.2109509:
                if input[5280] < 0.0730186:
                    var135 = 0.0413163
                else:
                    var135 = 0.1449055
            else:
                if input[3270] < 0.48053354:
                    var135 = 0.1014924
                else:
                    if input[5189] < 1.1187681:
                        if input[9679] < 0.11908884:
                            var135 = 0.13569213
                        else:
                            var135 = 0.038298182
                    else:
                        if input[9916] < 0.10695686:
                            var135 = 0.10958681
                        else:
                            var135 = -0.0028970274
    if input[8178] < 0.9718113:
        var136 = -0.11946382
    else:
        if input[3916] < 0.083009586:
            if input[9916] < 0.09787006:
                var136 = 0.021930672
            else:
                if input[6505] < 0.24528328:
                    var136 = 0.029441243
                else:
                    var136 = 0.11447396
        else:
            if input[6785] < 0.05536623:
                var136 = 0.10199862
            else:
                if input[6978] < 0.35193887:
                    var136 = 0.10417531
                else:
                    if input[5990] < 1.0:
                        if input[3311] >= 0.17861243:
                            var136 = 0.036161434
                        else:
                            var136 = 0.12915589
                    else:
                        if input[5869] >= 0.21954967:
                            var136 = -0.10226138
                        else:
                            var136 = -0.00076619297
    if input[6306] < 0.60670716:
        var137 = -0.10100833
    else:
        if input[7930] < 1.4311116:
            var137 = -0.12991565
        else:
            if input[4836] >= 0.24111131:
                var137 = -0.10087991
            else:
                if input[8584] >= 0.16258971:
                    if input[6785] >= 0.09394948:
                        if input[6785] < 0.10356217:
                            var137 = 0.09152793
                        else:
                            var137 = -0.020804483
                    else:
                        if input[3916] < 0.14016075:
                            var137 = -0.012920545
                        else:
                            var137 = -0.108912066
                else:
                    if input[4411] >= 0.18367273:
                        if input[4411] < 0.39986807:
                            var137 = -0.10948997
                        else:
                            var137 = 0.021181392
                    else:
                        if input[2762] < 0.22651482:
                            var137 = 0.09533824
                        else:
                            var137 = 0.0021000735
    if input[531] < 0.11878177:
        var138 = 0.12785213
    else:
        if input[903] < 1.5411538:
            if input[4170] >= 0.14478311:
                if input[4170] < 0.14731102:
                    var138 = -0.08216273
                else:
                    if input[903] < 0.24061747:
                        var138 = 0.08085402
                    else:
                        if input[4170] < 0.1996962:
                            var138 = -0.0010926928
                        else:
                            var138 = 0.04919443
            else:
                if input[6785] >= 0.08898466:
                    if input[903] < 0.2989472:
                        var138 = 0.05193689
                    else:
                        var138 = -0.018451964
                else:
                    if input[4921] < 0.31779823:
                        var138 = 0.029879147
                    else:
                        var138 = 0.120318554
        else:
            if input[1777] < 0.48005053:
                if input[2876] >= 0.25408772:
                    var138 = 0.041515823
                else:
                    var138 = 0.17808816
            else:
                if input[4255] < 0.50742763:
                    if input[8398] < 0.17346749:
                        var138 = -0.06282549
                    else:
                        if input[284] < 0.18539824:
                            var138 = -0.023264734
                        else:
                            var138 = 0.10768926
                else:
                    if input[7224] < 0.18973956:
                        if input[6785] >= 0.04730627:
                            var138 = 0.0785543
                        else:
                            var138 = 0.022574628
                    else:
                        if input[6534] >= 0.14375475:
                            var138 = -0.061027642
                        else:
                            var138 = -0.0018545149
    if input[8466] < 0.08764862:
        var139 = 0.11335321
    else:
        if input[6404] < 1.1448104:
            var139 = 0.12686032
        else:
            if input[9386] < 0.22473142:
                var139 = 0.099863894
            else:
                if input[962] < 0.51143813:
                    var139 = 0.11951293
                else:
                    if input[9006] < 0.6959612:
                        var139 = 0.10120984
                    else:
                        if input[710] >= 0.21040188:
                            var139 = 0.08876421
                        else:
                            var139 = -0.0026077586
    if input[9679] < 0.07169372:
        if input[6785] < 0.06442343:
            var140 = 0.004804572
        else:
            var140 = 0.10656647
    else:
        if input[3303] < 1.2570162:
            if input[6418] >= 0.20251761:
                var140 = 0.023330713
            else:
                if input[9679] < 0.12205061:
                    var140 = 0.03190124
                else:
                    var140 = 0.115213394
        else:
            if input[3236] >= 0.22634567:
                if input[3236] < 0.34863394:
                    var140 = 0.15628639
                else:
                    var140 = 0.046693005
            else:
                if input[2104] >= 0.21082489:
                    var140 = 0.10227736
                else:
                    if input[892] >= 0.19968705:
                        var140 = 0.1680241
                    else:
                        if input[6775] >= 0.14890413:
                            var140 = -0.102416016
                        else:
                            var140 = -0.0013986619
    if input[5378] < 0.18943173:
        if input[8239] < 0.17391537:
            if input[8239] < 0.09074553:
                var141 = 0.06144305
            else:
                var141 = -0.01070481
        else:
            if input[28] < 0.24747406:
                var141 = 0.021193923
            else:
                var141 = 0.11414305
    else:
        if input[2111] < 0.0937234:
            var141 = 0.1277713
        else:
            if input[1316] < 0.41825908:
                var141 = 0.12514833
            else:
                if input[5998] < 0.56336385:
                    var141 = 0.099526964
                else:
                    if input[3000] < 1.0769:
                        if input[4170] >= 0.1178779:
                            var141 = -0.09252334
                        else:
                            var141 = 0.050567236
                    else:
                        if input[4269] < 0.36174938:
                            var141 = 0.13019674
                        else:
                            var141 = -0.0030479142
    if input[4666] < 0.08900275:
        var142 = 0.13103847
    else:
        if input[3371] < 1.1404184:
            if input[7334] >= 0.20345515:
                if input[5280] >= 0.09922387:
                    var142 = 0.08490109
                else:
                    if input[6785] < 0.15161943:
                        var142 = 0.032316186
                    else:
                        if input[3371] < 0.26927283:
                            var142 = -0.10371734
                        else:
                            var142 = -0.0068674386
            else:
                if input[3381] >= 0.28113464:
                    var142 = -0.09625031
                else:
                    if input[2957] < 0.16803464:
                        var142 = -0.08358138
                    else:
                        if input[8239] >= 0.10258515:
                            var142 = -0.011056987
                        else:
                            var142 = 0.07796108
        else:
            if input[3504] < 1.4315313:
                if input[6785] < 0.16000375:
                    if input[5280] < 0.10509737:
                        var142 = 0.024578294
                    else:
                        var142 = -0.050094306
                else:
                    if input[8239] >= 0.10301286:
                        var142 = -0.012460164
                    else:
                        if input[5280] >= 0.14152125:
                            var142 = 0.052062936
                        else:
                            var142 = 0.15185264
            else:
                if input[5496] < 0.7983319:
                    if input[5496] < 0.13140815:
                        var142 = 0.10401205
                    else:
                        if input[3916] < 0.121714525:
                            var142 = -0.09784225
                        else:
                            var142 = 0.035393883
                else:
                    if input[6730] < 0.55022043:
                        if input[6785] >= 0.08058731:
                            var142 = 0.030165836
                        else:
                            var142 = 0.1268251
                    else:
                        if input[4104] < 0.27949:
                            var142 = 0.08355827
                        else:
                            var142 = -0.004238227
    if input[9048] < 1.1998109:
        var143 = 0.18425299
    else:
        if input[5531] >= 0.22028077:
            if input[6785] >= 0.1733538:
                var143 = 0.086986534
            else:
                if input[2957] < 0.16258192:
                    var143 = 0.0269902
                else:
                    if input[6785] < 0.11985802:
                        var143 = 0.007753666
                    else:
                        if input[5280] < 0.14704065:
                            var143 = -0.016669378
                        else:
                            var143 = -0.10441766
        else:
            if input[2679] < 1.4071828:
                if input[3609] < 0.16402929:
                    var143 = -0.015378731
                else:
                    if input[5189] >= 0.042941667:
                        var143 = 0.002976426
                    else:
                        if input[5531] >= 0.14607929:
                            var143 = 0.0047939634
                        else:
                            var143 = 0.10659331
            else:
                if input[710] >= 0.22767049:
                    var143 = 0.09256368
                else:
                    if input[8626] < 0.0879145:
                        var143 = 0.11871692
                    else:
                        if input[5463] < 1.0120678:
                            var143 = 0.10383721
                        else:
                            var143 = -0.0013523496
    if input[5736] < 0.44086555:
        if input[8239] < 0.105920464:
            var144 = -0.011011226
        else:
            if input[5736] < 0.17767654:
                if input[5736] < 0.15716292:
                    var144 = 0.07149815
                else:
                    var144 = -0.042050462
            else:
                if input[5113] >= 0.16126575:
                    var144 = 0.013202799
                else:
                    var144 = 0.103959516
    else:
        if input[2680] < 2.00001:
            if input[4666] < 0.14521445:
                var144 = -0.010527837
            else:
                if input[8239] >= 0.13896155:
                    var144 = 0.0015331535
                else:
                    if input[2680] < 0.25381976:
                        if input[2680] < 0.23268394:
                            var144 = 0.110663414
                        else:
                            var144 = -0.036729366
                    else:
                        var144 = 0.13190638
        else:
            if input[9991] < 0.3367947:
                if input[9916] >= 0.13737911:
                    var144 = -0.0045687547
                else:
                    if input[9679] < 0.22241081:
                        var144 = 0.051154505
                    else:
                        var144 = 0.16905338
            else:
                if input[2722] >= 0.18979347:
                    var144 = -0.105128065
                else:
                    if input[3892] < 0.58796895:
                        var144 = 0.17291114
                    else:
                        if input[5860] >= 0.14867797:
                            var144 = -0.09231055
                        else:
                            var144 = -0.00054711144
    if input[2782] >= 0.16067313:
        if input[9679] < 0.11667014:
            var145 = 0.016274683
        else:
            if input[4170] >= 0.13133019:
                var145 = -0.04639353
            else:
                var145 = -0.10938622
    else:
        if input[5358] < 0.337976:
            if input[5280] < 0.072184354:
                var145 = -0.053132027
            else:
                if input[5357] < 0.23418093:
                    if input[5280] < 0.107992105:
                        var145 = 0.12322388
                    else:
                        var145 = -0.022302324
                else:
                    var145 = 0.22080393
        else:
            if input[2111] < 0.09741214:
                var145 = 0.12920342
            else:
                if input[1664] < 0.22088127:
                    if input[6785] >= 0.08784678:
                        var145 = -0.04538503
                    else:
                        var145 = 0.102974705
                else:
                    if input[9162] < 0.24447975:
                        if input[9557] >= 0.16911274:
                            var145 = 0.016479505
                        else:
                            var145 = 0.11225646
                    else:
                        if input[7995] < 1.1403339:
                            var145 = 0.13540518
                        else:
                            var145 = -0.0015031038
    if input[118] < 1.1779219:
        var146 = 0.15602306
    else:
        if input[6418] < 0.34574562:
            if input[9307] < 0.16416317:
                if input[6418] < 0.20892265:
                    var146 = -0.06704583
                else:
                    var146 = -0.015114232
            else:
                if input[3214] >= 0.24440584:
                    var146 = -0.052235335
                else:
                    if input[6785] < 0.15889597:
                        var146 = -0.05043908
                    else:
                        if input[6423] >= 0.3641763:
                            var146 = -0.046848726
                        else:
                            var146 = 0.097537994
        else:
            if input[5476] < 0.20807692:
                if input[8239] < 0.11622336:
                    if input[9679] >= 0.097336516:
                        var146 = -0.11023676
                    else:
                        if input[5476] < 0.17586254:
                            var146 = 0.066339135
                        else:
                            var146 = 0.0004118104
                else:
                    if input[3844] >= 0.15722874:
                        var146 = -0.03205847
                    else:
                        if input[1350] < 0.18537076:
                            var146 = -0.033205133
                        else:
                            var146 = 0.102446854
            else:
                if input[1350] < 0.28775734:
                    if input[6785] < 0.1337869:
                        if input[1350] < 0.2212062:
                            var146 = -0.0015817899
                        else:
                            var146 = -0.09153396
                    else:
                        if input[8466] < 0.20623562:
                            var146 = -0.009389073
                        else:
                            var146 = 0.10888534
                else:
                    if input[8933] < 0.9507994:
                        if input[6785] >= 0.07740547:
                            var146 = -0.056032192
                        else:
                            var146 = 0.101984344
                    else:
                        if input[3641] < 0.23051533:
                            var146 = 0.074362926
                        else:
                            var146 = -0.0035095327
    var147 = var91 + var92 + var93 + var94 + var95 + var96 + var97 + var98 + var99 + var100 + var101 + var102 + var103 + var104 + var105 + var106 + var107 + var108 + var109 + var110 + var111 + var112 + var113 + var114 + var115 + var116 + var117 + var118 + var119 + var120 + var121 + var122 + var123 + var124 + var125 + var126 + var127 + var128 + var129 + var130 + var131 + var132 + var133 + var134 + var135 + var136 + var137 + var138 + var139 + var140 + var141 + var142 + var143 + var144 + var145 + var146
    if input[8148] < 1.5330952:
        if input[8148] < 0.39003524:
            var148 = 0.0614988
        else:
            var148 = 0.23204641
    else:
        if input[5248] < 0.37926525:
            if input[3844] < 0.18173622:
                var148 = 0.016927108
            else:
                var148 = 0.14278112
        else:
            if input[3270] < 0.48053354:
                var148 = 0.1003502
            else:
                if input[2491] < 0.58572185:
                    if input[9679] < 0.13211595:
                        var148 = 0.005046125
                    else:
                        if input[9307] >= 0.21131058:
                            var148 = 0.0003009832
                        else:
                            var148 = 0.1196061
                else:
                    if input[22] < 0.7892361:
                        if input[3844] < 0.12472407:
                            var148 = 0.01132143
                        else:
                            var148 = 0.13435112
                    else:
                        if input[4952] < 0.85737866:
                            var148 = 0.15885045
                        else:
                            var148 = -0.0020635817
    if input[425] >= 0.34232843:
        var149 = 0.17741536
    else:
        if input[9525] < 1.241941:
            var149 = 0.12534262
        else:
            if input[5280] < 0.048058297:
                var149 = 0.10189619
            else:
                if input[5369] >= 0.21492615:
                    if input[6785] >= 0.106665775:
                        var149 = -0.00552172
                    else:
                        if input[5280] < 0.122210816:
                            var149 = 0.19595282
                        else:
                            var149 = 0.0670222
                else:
                    if input[6310] >= 0.2118919:
                        if input[6312] < 0.3387697:
                            var149 = 0.021586886
                        else:
                            var149 = 0.16843258
                    else:
                        if input[4878] >= 0.2709588:
                            var149 = -0.1182462
                        else:
                            var149 = -0.001171534
    if input[8792] >= 0.13564949:
        if input[6785] >= 0.10403197:
            var150 = 0.008061059
        else:
            if input[8792] < 0.19012164:
                var150 = -0.09826955
            else:
                var150 = -0.033887245
    else:
        if input[8923] >= 0.17807269:
            if input[4012] >= 0.1563553:
                var150 = 0.022214996
            else:
                if input[5113] < 0.16925263:
                    var150 = 0.011259956
                else:
                    if input[5587] < 0.85903096:
                        var150 = -0.0031447464
                    else:
                        if input[6785] < 0.14139235:
                            var150 = -0.038125735
                        else:
                            var150 = -0.11082486
        else:
            if input[5422] >= 0.19210783:
                if input[1347] < 0.46965322:
                    var150 = 0.036147498
                else:
                    if input[1812] < 0.24573593:
                        var150 = -0.001388885
                    else:
                        if input[8787] >= 0.22351204:
                            var150 = -0.014537017
                        else:
                            var150 = -0.110476196
            else:
                if input[3878] < 0.3738104:
                    if input[3880] >= 0.2621809:
                        var150 = -0.069360994
                    else:
                        if input[3844] >= 0.16662619:
                            var150 = -0.068878315
                        else:
                            var150 = 0.114043556
                else:
                    if input[8219] < 0.38585815:
                        if input[8466] >= 0.12111029:
                            var150 = -0.0349367
                        else:
                            var150 = 0.11926112
                    else:
                        if input[3892] < 0.58796895:
                            var150 = 0.16687351
                        else:
                            var150 = 0.0004714527
    if input[5105] >= 0.1816007:
        if input[6785] >= 0.114068605:
            var151 = 0.010906191
        else:
            var151 = -0.11129873
    else:
        if input[6386] < 0.51393956:
            if input[6319] < 0.33864787:
                if input[6386] < 0.27865097:
                    var151 = -0.050156098
                else:
                    var151 = 0.059185605
            else:
                var151 = -0.13605805
        else:
            if input[4908] < 2.00001:
                var151 = -0.12078809
            else:
                if input[6306] < 0.60670716:
                    var151 = -0.09936855
                else:
                    if input[3916] >= 0.2317871:
                        if input[8239] >= 0.103680596:
                            var151 = 0.06775301
                        else:
                            var151 = -0.089605354
                    else:
                        if input[212] >= 0.21143489:
                            var151 = -0.09354928
                        else:
                            var151 = 0.0029847294
    if input[1779] < 1.2002537:
        if input[1781] < 0.52772534:
            if input[1781] < 0.3292537:
                var152 = 0.08992984
            else:
                var152 = -0.042928316
        else:
            var152 = 0.14796792
    else:
        if input[4068] >= 0.15810794:
            if input[4076] >= 0.27483052:
                if input[4068] < 0.21971458:
                    var152 = -0.03698379
                else:
                    var152 = 0.054744124
            else:
                if input[5280] >= 0.11019604:
                    if input[4068] < 0.26972327:
                        var152 = 0.016540995
                    else:
                        var152 = -0.011047776
                else:
                    if input[8626] >= 0.13634428:
                        var152 = 0.016425699
                    else:
                        if input[6785] >= 0.11370214:
                            var152 = 0.016132025
                        else:
                            var152 = -0.10215396
        else:
            if input[7334] < 0.20345515:
                if input[5819] < 0.17430775:
                    var152 = -0.037464403
                else:
                    if input[3916] >= 0.097525634:
                        var152 = -0.03508925
                    else:
                        if input[8239] < 0.095451854:
                            var152 = -0.010988492
                        else:
                            var152 = 0.103831194
            else:
                if input[3256] >= 0.17889965:
                    if input[2910] < 0.2501131:
                        var152 = -0.019006737
                    else:
                        var152 = -0.12211173
                else:
                    if input[6321] < 0.2527963:
                        if input[9379] < 0.16865161:
                            var152 = 0.10028421
                        else:
                            var152 = 0.024383737
                    else:
                        if input[903] < 1.5411538:
                            var152 = 0.07941512
                        else:
                            var152 = -0.0008739076
    if input[2371] < 1.2550216:
        if input[9491] >= 0.2358665:
            var153 = -0.006204471
        else:
            var153 = 0.14179562
    else:
        if input[911] >= 0.2729628:
            var153 = 0.1898993
        else:
            if input[6465] < 0.5095433:
                if input[5459] < 0.3057792:
                    var153 = -0.03966373
                else:
                    if input[6785] >= 0.12599452:
                        if input[6465] < 0.32386243:
                            var153 = -0.026496014
                        else:
                            var153 = 0.02916014
                    else:
                        if input[9916] >= 0.12261604:
                            var153 = -0.013713531
                        else:
                            var153 = 0.1142685
            else:
                if input[5459] < 0.28701678:
                    if input[5496] < 0.18194908:
                        var153 = 0.0057503385
                    else:
                        if input[9379] >= 0.12148764:
                            var153 = -0.001381331
                        else:
                            var153 = 0.12399459
                else:
                    if input[4513] >= 0.23678838:
                        var153 = 0.09653957
                    else:
                        if input[9006] < 0.6959612:
                            var153 = 0.10003189
                        else:
                            var153 = -0.0024671608
    if input[4569] < 2.00001:
        var154 = 0.20498557
    else:
        if input[5990] < 1.0:
            var154 = 0.11922722
        else:
            if input[9307] < 0.09377032:
                var154 = 0.1251627
            else:
                if input[8094] >= 0.1990066:
                    if input[9916] < 0.23533554:
                        var154 = -0.06731757
                    else:
                        if input[8239] < 0.15498516:
                            var154 = -0.036178354
                        else:
                            var154 = 0.09787335
                else:
                    if input[2997] < 0.67106825:
                        if input[2957] < 0.19303094:
                            var154 = 0.0039039575
                        else:
                            var154 = 0.09088789
                    else:
                        if input[4878] >= 0.2709588:
                            var154 = -0.113455474
                        else:
                            var154 = -0.0014972644
    if input[8424] < 0.13282582:
        if input[8239] >= 0.06659564:
            var155 = 0.020151338
        else:
            var155 = 0.12400458
    else:
        if input[4012] < 0.101977214:
            if input[4012] < 0.085529275:
                var155 = 0.13350797
            else:
                var155 = 0.054171927
        else:
            if input[2104] >= 0.21082489:
                var155 = 0.101026945
            else:
                if input[963] < 0.7767438:
                    if input[4170] >= 0.11331123:
                        var155 = -0.03940298
                    else:
                        if input[5322] >= 0.17461243:
                            var155 = 0.019768117
                        else:
                            var155 = 0.119929604
                else:
                    if input[8218] < 1.0324993:
                        var155 = 0.13001275
                    else:
                        if input[5395] < 0.8492508:
                            var155 = 0.14335471
                        else:
                            var155 = -0.0022111263
    if input[8398] < 0.12340101:
        if input[8466] >= 0.033021607:
            var156 = 0.028365284
        else:
            var156 = 0.11993575
    else:
        if input[6404] < 1.1448104:
            var156 = 0.122341454
        else:
            if input[3889] < 1.2113615:
                var156 = 0.18725096
            else:
                if input[3269] < 0.20942095:
                    if input[5280] >= 0.07086958:
                        if input[4170] >= 0.13058949:
                            var156 = 0.030803949
                        else:
                            var156 = -0.078644276
                    else:
                        if input[8239] < 0.11154473:
                            var156 = -0.03483978
                        else:
                            var156 = 0.08494778
                else:
                    if input[8119] >= 0.19171068:
                        if input[5113] < 0.17682132:
                            var156 = 0.006243402
                        else:
                            var156 = -0.10364598
                    else:
                        if input[6522] < 0.28518876:
                            var156 = 0.08478009
                        else:
                            var156 = -0.001521429
    if input[2033] < 0.3023872:
        if input[809] < 0.19113877:
            var157 = -0.09943443
        else:
            var157 = -0.026532665
    else:
        if input[9274] >= 0.21646078:
            var157 = -0.09908155
        else:
            if input[9001] >= 0.23266466:
                if input[5280] >= 0.09236117:
                    var157 = -0.02096094
                else:
                    if input[3916] >= 0.12487771:
                        var157 = -0.033706296
                    else:
                        var157 = -0.13014914
            else:
                if input[2658] < 1.2680178:
                    if input[9916] >= 0.13010333:
                        var157 = -0.06328864
                    else:
                        if input[749] >= 0.1697041:
                            var157 = -0.042993855
                        else:
                            var157 = 0.09492346
                else:
                    if input[9557] < 0.3245704:
                        if input[9679] >= 0.088964455:
                            var157 = -0.05843354
                        else:
                            var157 = 0.093475044
                    else:
                        if input[5935] < 0.9499705:
                            var157 = 0.17959248
                        else:
                            var157 = -0.00008031999
    if input[5998] < 0.56336385:
        var158 = 0.098513626
    else:
        if input[2674] < 1.8983641:
            var158 = 0.1647232
        else:
            if input[2677] < 2.00001:
                if input[5189] >= 0.17032836:
                    var158 = -0.038394753
                else:
                    if input[8239] < 0.121802166:
                        var158 = -0.001663613
                    else:
                        if input[5531] >= 0.04742822:
                            var158 = -0.0068329163
                        else:
                            var158 = 0.10846895
            else:
                if input[2387] < 0.6178757:
                    if input[8466] < 0.16444744:
                        var158 = 0.020475972
                    else:
                        var158 = 0.1279962
                else:
                    if input[3303] < 1.2570162:
                        if input[6418] >= 0.1986684:
                            var158 = 0.03274807
                        else:
                            var158 = 0.1088892
                    else:
                        if input[595] < 0.55562544:
                            var158 = 0.10416155
                        else:
                            var158 = -0.0024113536
    if input[5531] < 0.10104962:
        if input[5280] < 0.052636597:
            var159 = 0.015193753
        else:
            var159 = 0.13847686
    else:
        if input[5736] >= 0.18245144:
            if input[3000] >= 0.1775539:
                var159 = 0.010604161
            else:
                if input[9468] < 0.14861166:
                    var159 = 0.013050462
                else:
                    if input[5113] >= 0.17151158:
                        var159 = 0.018248927
                    else:
                        var159 = 0.10358962
        else:
            if input[749] < 0.13306631:
                var159 = 0.124159865
            else:
                if input[6611] < 0.32068118:
                    if input[3825] < 0.1726117:
                        var159 = -0.054813027
                    else:
                        if input[8416] < 0.42093498:
                            var159 = -0.037475523
                        else:
                            var159 = 0.08799227
                else:
                    if input[4170] >= 0.21689074:
                        if input[316] >= 0.32117757:
                            var159 = -0.0591057
                        else:
                            var159 = 0.061197173
                    else:
                        if input[1777] >= 0.22549634:
                            var159 = 0.14637285
                        else:
                            var159 = -0.0027666453
    if input[7930] < 1.4311116:
        var160 = -0.1231293
    else:
        if input[9468] < 0.16136196:
            if input[9213] >= 0.11924965:
                if input[8239] < 0.10935012:
                    var160 = -0.109365
                else:
                    var160 = -0.015869295
            else:
                if input[8178] < 0.25306106:
                    var160 = -0.08089581
                else:
                    if input[8185] >= 0.17975031:
                        var160 = -0.08754314
                    else:
                        if input[8626] < 0.11962835:
                            var160 = -0.05112685
                        else:
                            var160 = 0.07579692
        else:
            if input[6326] < 0.30980593:
                if input[8239] >= 0.13470574:
                    var160 = -0.036166776
                else:
                    if input[3311] >= 0.11520159:
                        var160 = -0.017277991
                    else:
                        if input[3844] < 0.20087151:
                            var160 = 0.035095733
                        else:
                            var160 = 0.111777626
            else:
                if input[7334] < 0.53020924:
                    if input[5280] >= 0.11687931:
                        if input[7334] < 0.28172037:
                            var160 = -0.09499391
                        else:
                            var160 = 0.023411319
                    else:
                        if input[7376] >= 0.166726:
                            var160 = -0.0818403
                        else:
                            var160 = 0.06423336
                else:
                    if input[5395] < 0.8492508:
                        var160 = 0.14751579
                    else:
                        if input[248] < 0.21890488:
                            var160 = 0.091512986
                        else:
                            var160 = -0.0027351433
    if input[3891] < 1.3429911:
        var161 = 0.15686092
    else:
        if input[908] < 1.8546642:
            if input[908] < 0.37637743:
                var161 = 0.15165828
            else:
                var161 = 0.05411468
        else:
            if input[4326] < 1.6494892:
                if input[4326] < 0.36764124:
                    if input[4326] < 0.24344745:
                        if input[8466] >= 0.11813959:
                            var161 = 0.03718766
                        else:
                            var161 = 0.14174183
                    else:
                        if input[4326] < 0.31650817:
                            var161 = 0.033308614
                        else:
                            var161 = -0.06963594
                else:
                    var161 = 0.15542816
            else:
                if input[4607] < 0.26056585:
                    if input[6785] < 0.14548054:
                        var161 = 0.04637608
                    else:
                        if input[4607] < 0.15140015:
                            var161 = 0.00006276485
                        else:
                            var161 = -0.096428655
                else:
                    if input[2920] < 1.1525533:
                        if input[2920] < 0.4299407:
                            var161 = 0.15182428
                        else:
                            var161 = 0.043982696
                    else:
                        if input[2562] >= 0.17829482:
                            var161 = 0.0729741
                        else:
                            var161 = -0.0011518594
    if input[3087] >= 0.22436564:
        if input[5280] < 0.1485537:
            if input[3087] < 0.26384592:
                if input[3087] < 0.24153394:
                    var162 = -0.020236744
                else:
                    var162 = -0.0780302
            else:
                var162 = 0.044257443
        else:
            if input[8626] < 0.15942161:
                var162 = -0.011418163
            else:
                if input[4012] >= 0.1710129:
                    var162 = -0.030334806
                else:
                    var162 = -0.11381614
    else:
        if input[1239] >= 0.17618169:
            if input[6785] >= 0.11051428:
                var162 = 0.007396837
            else:
                var162 = 0.09846436
        else:
            if input[1169] >= 0.21412548:
                if input[5280] < 0.19833048:
                    var162 = 0.011792192
                else:
                    var162 = 0.1000661
            else:
                if input[809] >= 0.1646886:
                    if input[2111] >= 0.14846742:
                        var162 = 0.042724036
                    else:
                        if input[8239] < 0.14557505:
                            var162 = 0.021182736
                        else:
                            var162 = -0.10562885
                else:
                    if input[5997] >= 0.14492221:
                        if input[9006] < 0.6959612:
                            var162 = 0.09886461
                        else:
                            var162 = -0.10567937
                    else:
                        if input[7553] < 0.59866923:
                            var162 = 0.09466117
                        else:
                            var162 = 0.0005916475
    if input[5372] < 1.2109509:
        if input[5280] < 0.09770191:
            var163 = 0.05948654
        else:
            var163 = 0.14431606
    else:
        if input[5819] < 0.122899115:
            if input[5280] < 0.07902782:
                var163 = 0.020864055
            else:
                var163 = 0.1294009
        else:
            if input[3236] >= 0.22634567:
                if input[3236] < 0.34863394:
                    var163 = 0.14470135
                else:
                    var163 = 0.044678945
            else:
                if input[6978] < 0.35193887:
                    var163 = 0.10165345
                else:
                    if input[5113] < 0.100534074:
                        var163 = 0.12474056
                    else:
                        if input[892] >= 0.19968705:
                            var163 = 0.15381691
                        else:
                            var163 = -0.0019062379
    if input[3247] >= 0.19587226:
        if input[8466] < 0.12262777:
            var164 = -0.018961627
        else:
            if input[6785] < 0.09670448:
                var164 = -0.022641305
            else:
                var164 = -0.11093147
    else:
        if input[3886] < 0.6754467:
            var164 = 0.15770043
        else:
            if input[3371] < 0.17540549:
                if input[7334] >= 0.17139219:
                    var164 = -0.015135631
                else:
                    if input[3916] < 0.10589841:
                        var164 = 0.00027210926
                    else:
                        if input[8590] >= 0.13623562:
                            var164 = 0.0043832497
                        else:
                            var164 = 0.115285955
            else:
                if input[3641] < 0.31751505:
                    if input[8626] < 0.2032789:
                        if input[6785] < 0.123292536:
                            var164 = 0.054123234
                        else:
                            var164 = -0.08373125
                    else:
                        if input[2201] < 0.2552148:
                            var164 = -0.0676805
                        else:
                            var164 = 0.087431364
                else:
                    if input[3609] < 0.3483532:
                        if input[6785] < 0.22547413:
                            var164 = 0.037786372
                        else:
                            var164 = -0.052277334
                    else:
                        if input[284] >= 0.1436193:
                            var164 = -0.045518138
                        else:
                            var164 = 0.0015705243
    if input[881] < 1.0:
        if input[9679] >= 0.13058922:
            var165 = 0.037819713
        else:
            var165 = 0.12447422
    else:
        if input[5860] >= 0.14867797:
            if input[5860] < 0.27757886:
                if input[5860] < 0.24390705:
                    if input[5860] < 0.22475342:
                        var165 = -0.076228544
                    else:
                        var165 = 0.018628007
                else:
                    var165 = -0.09838425
            else:
                var165 = -0.017880779
        else:
            if input[1812] >= 0.20596889:
                if input[4012] < 0.17289436:
                    var165 = 0.10005262
                else:
                    if input[1381] < 0.38422158:
                        var165 = 0.027202321
                    else:
                        if input[2691] >= 0.30450696:
                            var165 = 0.0034833055
                        else:
                            var165 = -0.102411024
            else:
                if input[1321] < 0.47067758:
                    if input[6785] < 0.127921:
                        if input[1321] < 0.25102943:
                            var165 = -0.04414666
                        else:
                            var165 = 0.017667754
                    else:
                        if input[3825] >= 0.23903799:
                            var165 = -0.018085629
                        else:
                            var165 = 0.11548787
                else:
                    if input[5725] < 0.22910857:
                        if input[9916] >= 0.11777075:
                            var165 = -0.04806978
                        else:
                            var165 = 0.110453226
                    else:
                        if input[4012] >= 0.109106086:
                            var165 = -0.03089953
                        else:
                            var165 = 0.0013613704
    if input[2221] < 1.1485573:
        if input[8239] >= 0.09812241:
            var166 = -0.062450837
        else:
            if input[5280] < 0.21638642:
                if input[6252] >= 0.18988906:
                    var166 = 0.03673952
                else:
                    var166 = 0.15770574
            else:
                if input[6785] < 0.16795206:
                    var166 = 0.09008078
                else:
                    if input[2221] < 0.19559659:
                        var166 = 0.06693444
                    else:
                        if input[9679] < 0.20486479:
                            var166 = 0.04255233
                        else:
                            var166 = -0.04776695
    else:
        if input[9048] < 1.1998109:
            var166 = 0.17056762
        else:
            if input[8239] < 0.07834077:
                if input[9679] < 0.07362322:
                    var166 = 0.005166651
                else:
                    var166 = 0.10271835
            else:
                if input[9679] < 0.07169372:
                    var166 = 0.10349399
                else:
                    if input[1316] < 0.41825908:
                        var166 = 0.11786356
                    else:
                        if input[2104] >= 0.21082489:
                            var166 = 0.09979423
                        else:
                            var166 = -0.0021807046
    if input[8178] < 0.9718113:
        var167 = -0.11138747
    else:
        if input[6785] < 0.05536623:
            var167 = 0.10005122
        else:
            if input[2680] < 2.00001:
                if input[2680] < 0.25219274:
                    if input[2680] < 0.23268394:
                        var167 = 0.079755686
                    else:
                        var167 = -0.034891464
                else:
                    if input[8239] < 0.15147093:
                        var167 = 0.030802399
                    else:
                        var167 = 0.12685333
            else:
                if input[9991] < 0.3367947:
                    if input[9679] >= 0.12006375:
                        var167 = -0.00046771902
                    else:
                        if input[3844] >= 0.15379576:
                            var167 = 0.01901867
                        else:
                            var167 = 0.16123116
                else:
                    if input[2722] >= 0.18979347:
                        var167 = -0.100577764
                    else:
                        if input[5736] < 0.44086555:
                            var167 = 0.08316671
                        else:
                            var167 = -0.0008056437
    if input[718] < 0.48527673:
        if input[718] < 0.28119665:
            if input[8239] < 0.17059773:
                var168 = 0.075786956
            else:
                var168 = -0.012673125
        else:
            var168 = 0.14491878
    else:
        if input[3908] < 0.89272755:
            if input[8239] < 0.1569696:
                var168 = -0.014564029
            else:
                if input[6785] >= 0.08635712:
                    var168 = -0.018466063
                else:
                    if input[5476] >= 0.17067823:
                        var168 = 0.010034641
                    else:
                        var168 = 0.14921483
        else:
            if input[2373] < 1.2921082:
                var168 = 0.13384713
            else:
                if input[5280] < 0.048058297:
                    var168 = 0.10021808
                else:
                    if input[710] >= 0.21040188:
                        var168 = 0.085775755
                    else:
                        if input[6785] < 2.00001:
                            var168 = 0.011724865
                        else:
                            var168 = -0.0051084897
    if input[5378] < 0.18943173:
        if input[8239] < 0.17391537:
            var169 = 0.026349276
        else:
            var169 = 0.10407237
    else:
        if input[8584] >= 0.16258971:
            if input[6785] < 0.15035632:
                if input[9679] < 0.12505844:
                    var169 = 0.109074466
                else:
                    var169 = -0.026871396
            else:
                if input[3916] < 0.14016075:
                    if input[8584] < 0.20617667:
                        var169 = -0.021161182
                    else:
                        var169 = 0.009310211
                else:
                    if input[8586] < 0.8809196:
                        var169 = -0.0005144795
                    else:
                        if input[3609] >= 0.13426167:
                            var169 = -0.009180329
                        else:
                            var169 = -0.11783663
        else:
            if input[5869] >= 0.21954967:
                var169 = -0.09508123
            else:
                if input[2782] >= 0.16067313:
                    if input[9679] < 0.11667014:
                        var169 = 0.016416406
                    else:
                        if input[4170] >= 0.13133019:
                            var169 = -0.04382034
                        else:
                            var169 = -0.1046945
                else:
                    if input[7995] < 1.1403339:
                        if input[9679] >= 0.12524232:
                            var169 = -0.004926366
                        else:
                            var169 = 0.15487023
                    else:
                        if input[3762] >= 0.16158153:
                            var169 = -0.09682322
                        else:
                            var169 = 0.0013192716
    if input[531] < 0.11878177:
        var170 = 0.12222482
    else:
        if input[962] < 0.51143813:
            var170 = 0.111353375
        else:
            if input[3504] < 1.4315313:
                if input[6785] < 0.16000375:
                    if input[5280] < 0.10509737:
                        var170 = 0.020922678
                    else:
                        var170 = -0.031973023
                else:
                    if input[6213] >= 0.32411885:
                        var170 = 0.006112397
                    else:
                        if input[5280] < 0.13801873:
                            var170 = 0.16224793
                        else:
                            var170 = 0.06820678
            else:
                if input[1779] < 1.2002537:
                    if input[1781] < 0.52772534:
                        if input[1781] < 0.3292537:
                            var170 = 0.08576956
                        else:
                            var170 = -0.04064754
                    else:
                        var170 = 0.1392819
                else:
                    if input[5248] < 0.37926525:
                        if input[5463] >= 0.24070011:
                            var170 = 0.018102488
                        else:
                            var170 = 0.13398194
                    else:
                        if input[2932] < 1.0769625:
                            var170 = 0.13299203
                        else:
                            var170 = -0.0020173565
    if input[8148] < 1.5330952:
        if input[8148] < 0.39003524:
            var171 = 0.058131497
        else:
            var171 = 0.20442061
    else:
        if input[5694] < 1.1638609:
            if input[284] < 0.26013443:
                if input[284] < 0.15289518:
                    var171 = 0.034878746
                else:
                    if input[4255] >= 0.22980176:
                        var171 = -0.013844858
                    else:
                        var171 = -0.07533405
            else:
                if input[4261] < 1.070682:
                    var171 = -0.06529159
                else:
                    if input[4259] < 0.2121602:
                        var171 = -0.012961161
                    else:
                        if input[4259] >= 0.28400993:
                            var171 = 0.010216749
                        else:
                            var171 = 0.114571415
        else:
            if input[8398] < 0.12340101:
                if input[8466] >= 0.033021607:
                    var171 = 0.026911283
                else:
                    var171 = 0.123039484
            else:
                if input[9916] < 0.10695686:
                    if input[4666] >= 0.07269929:
                        var171 = -0.053830136
                    else:
                        if input[3916] >= 0.070479095:
                            var171 = 0.035361987
                        else:
                            var171 = 0.12230398
                else:
                    if input[9307] < 0.11202816:
                        if input[9307] < 0.10403297:
                            var171 = 0.124908246
                        else:
                            var171 = 0.0389691
                    else:
                        if input[911] < 1.1692451:
                            var171 = 0.15531848
                        else:
                            var171 = -0.002110987
    if input[9525] < 1.241941:
        var172 = 0.12055832
    else:
        if input[425] >= 0.31366408:
            var172 = 0.15054232
        else:
            if input[4666] < 0.08900275:
                var172 = 0.12533598
            else:
                if input[275] < 0.47358245:
                    var172 = 0.11509936
                else:
                    if input[5463] < 1.0120678:
                        if input[8466] < 0.15199773:
                            var172 = -0.0154698035
                        else:
                            var172 = 0.10774969
                    else:
                        if input[3000] < 1.0769:
                            var172 = 0.040129464
                        else:
                            var172 = -0.0024854846
    if input[6386] < 0.51393956:
        if input[6319] < 0.33864787:
            var173 = 0.005946949
        else:
            var173 = -0.12921119
    else:
        if input[8224] < 0.40303355:
            if input[3825] >= 0.14603283:
                var173 = -0.081412815
            else:
                if input[6785] >= 0.09359331:
                    var173 = 0.0035168782
                else:
                    if input[9379] < 0.1759417:
                        var173 = 0.017296562
                    else:
                        var173 = 0.16094416
        else:
            if input[2344] < 0.2415315:
                var173 = 0.0961282
            else:
                if input[5189] < 1.1187681:
                    if input[9679] < 0.11908884:
                        var173 = 0.12795727
                    else:
                        if input[5189] < 0.36288875:
                            var173 = 0.026518045
                        else:
                            var173 = 0.13980383
                else:
                    if input[2677] < 2.00001:
                        if input[8239] < 0.121802166:
                            var173 = -0.003385214
                        else:
                            var173 = 0.096479505
                    else:
                        if input[5725] < 0.23397385:
                            var173 = 0.09432797
                        else:
                            var173 = -0.0023136085
    if input[8626] < 0.0879145:
        if input[5280] < 0.06235251:
            var174 = 0.039784428
        else:
            var174 = 0.12023727
    else:
        if input[6404] < 1.1448104:
            var174 = 0.11828567
        else:
            if input[8094] >= 0.1990066:
                if input[8239] < 0.15498516:
                    var174 = -0.03324044
                else:
                    if input[6785] < 0.12869906:
                        var174 = 0.0144959465
                    else:
                        var174 = 0.08624127
            else:
                if input[8792] >= 0.13564949:
                    if input[6785] >= 0.10403197:
                        var174 = 0.010961743
                    else:
                        var174 = -0.097561084
                else:
                    if input[3155] < 0.18472092:
                        if input[9679] < 0.10883236:
                            var174 = 0.034303628
                        else:
                            var174 = 0.15311961
                    else:
                        if input[5422] >= 0.1981499:
                            var174 = -0.0865555
                        else:
                            var174 = -0.00037293413
    if input[6306] < 0.60670716:
        var175 = -0.09761534
    else:
        if input[4908] < 2.00001:
            var175 = -0.11563455
        else:
            if input[8923] >= 0.17807269:
                if input[4012] >= 0.1563553:
                    var175 = 0.024396474
                else:
                    if input[6785] < 0.14139235:
                        if input[8923] < 0.2395259:
                            var175 = -0.0036151067
                        else:
                            var175 = -0.038185265
                    else:
                        if input[5587] < 0.85903096:
                            var175 = 0.015458322
                        else:
                            var175 = -0.103909455
            else:
                if input[8236] < 0.42157084:
                    if input[5113] >= 0.14305022:
                        var175 = -0.056766767
                    else:
                        if input[9307] >= 0.10857439:
                            var175 = 0.20434414
                        else:
                            var175 = 0.08623897
                else:
                    if input[710] >= 0.21040188:
                        var175 = 0.09061458
                    else:
                        if input[2679] < 1.4071828:
                            var175 = 0.08213851
                        else:
                            var175 = 0.00005295626
    if input[4569] < 2.00001:
        var176 = 0.18547088
    else:
        if input[4952] < 0.85737866:
            if input[4953] >= 0.15891828:
                var176 = 0.04660524
            else:
                var176 = 0.15930365
        else:
            if input[4411] >= 0.18367273:
                if input[4411] < 0.39986807:
                    if input[4411] < 0.2231409:
                        if input[4411] < 0.2037996:
                            var176 = -0.106682114
                        else:
                            var176 = 0.06085051
                    else:
                        var176 = -0.12202698
                else:
                    var176 = 0.023319853
            else:
                if input[2491] < 0.58572185:
                    if input[9679] < 0.13211595:
                        var176 = -0.0010646743
                    else:
                        if input[8466] >= 0.12851535:
                            var176 = 0.0037432655
                        else:
                            var176 = 0.11463356
                else:
                    if input[22] < 0.7892361:
                        if input[3844] < 0.12472407:
                            var176 = 0.011134751
                        else:
                            var176 = 0.12833397
                    else:
                        if input[9557] < 0.20949642:
                            var176 = 0.10941207
                        else:
                            var176 = -0.0010628644
    if input[3270] < 0.48053354:
        var177 = 0.097825885
    else:
        if input[118] < 1.1779219:
            var177 = 0.14174347
        else:
            if input[2722] >= 0.18979347:
                var177 = -0.09853212
            else:
                if input[5358] < 0.337976:
                    if input[9679] < 0.22241081:
                        var177 = -0.077570565
                    else:
                        if input[2111] < 0.14932132:
                            var177 = -0.055221047
                        else:
                            var177 = 0.1478978
                else:
                    if input[2111] < 0.09741214:
                        var177 = 0.123853244
                    else:
                        if input[1664] < 0.22088127:
                            var177 = 0.08546349
                        else:
                            var177 = -0.0013374296
    if input[858] >= 0.1819202:
        var178 = -0.13239157
    else:
        if input[6619] < 0.34485063:
            if input[6619] < 0.17646419:
                var178 = 0.035822615
            else:
                var178 = 0.14633432
        else:
            if input[2221] < 1.1485573:
                if input[8239] >= 0.09812241:
                    var178 = -0.059865594
                else:
                    if input[5280] < 0.21638642:
                        if input[963] >= 0.2619306:
                            var178 = 0.037915185
                        else:
                            var178 = 0.15056393
                    else:
                        if input[2221] < 0.25882137:
                            var178 = 0.0829367
                        else:
                            var178 = 0.003997319
            else:
                if input[6838] < 0.2826712:
                    if input[8626] < 0.15823108:
                        var178 = -0.02559995
                    else:
                        if input[8466] >= 0.1303116:
                            var178 = -0.018464204
                        else:
                            var178 = 0.1021734
                else:
                    if input[8933] >= 0.18994792:
                        if input[6785] >= 0.09006323:
                            var178 = -0.048340563
                        else:
                            var178 = 0.101419725
                    else:
                        if input[3303] < 1.2570162:
                            var178 = 0.10198462
                        else:
                            var178 = -0.0017603853
    if input[3844] < 0.12264583:
        if input[8239] >= 0.086112104:
            var179 = 0.022100534
        else:
            var179 = 0.10078069
    else:
        if input[1169] >= 0.21412548:
            if input[5280] >= 0.108336665:
                var179 = -0.0059080846
            else:
                var179 = 0.09715731
        else:
            if input[809] >= 0.1646886:
                if input[2111] >= 0.14846742:
                    var179 = 0.04056188
                else:
                    if input[8239] < 0.14557505:
                        var179 = 0.020474853
                    else:
                        if input[6785] >= 0.11830796:
                            var179 = -0.015155447
                        else:
                            var179 = -0.10743324
            else:
                if input[5496] < 0.7983319:
                    if input[3844] < 0.21633394:
                        if input[5496] < 0.16536017:
                            var179 = 0.018351631
                        else:
                            var179 = -0.07484814
                    else:
                        if input[8424] < 0.1858038:
                            var179 = -0.045520525
                        else:
                            var179 = 0.05030797
                else:
                    if input[6730] < 0.55022043:
                        if input[6785] >= 0.08058731:
                            var179 = 0.024403857
                        else:
                            var179 = 0.11124797
                    else:
                        if input[5369] >= 0.21492615:
                            var179 = 0.13841395
                        else:
                            var179 = -0.0020402465
    if input[3256] >= 0.17889965:
        if input[2910] < 0.2647588:
            var180 = -0.024315365
        else:
            var180 = -0.10812712
    else:
        if input[5105] >= 0.1816007:
            if input[6785] >= 0.114068605:
                var180 = 0.010501369
            else:
                var180 = -0.10770341
        else:
            if input[8083] >= 0.12325554:
                if input[3825] < 0.21786834:
                    var180 = -0.020452341
                else:
                    var180 = -0.13816826
            else:
                if input[5469] < 0.3876029:
                    if input[5496] < 0.17906852:
                        var180 = 0.012231967
                    else:
                        if input[5476] >= 0.1661409:
                            var180 = 0.018489726
                        else:
                            var180 = 0.12760592
                else:
                    if input[903] < 1.5411538:
                        if input[4170] >= 0.14530462:
                            var180 = 0.018019259
                        else:
                            var180 = 0.09393795
                    else:
                        if input[1777] < 0.48005053:
                            var180 = 0.13470356
                        else:
                            var180 = 0.00005362732
    if input[8218] < 1.0324993:
        var181 = 0.12124739
    else:
        if input[3889] < 1.2113615:
            var181 = 0.17160785
        else:
            if input[7172] < 0.30559668:
                if input[7172] < 0.28208363:
                    var181 = 0.13525258
                else:
                    var181 = 0.039212182
            else:
                if input[3247] >= 0.18947433:
                    if input[8466] < 0.12262777:
                        var181 = -0.01930791
                    else:
                        if input[6785] < 0.09670448:
                            var181 = -0.02177567
                        else:
                            var181 = -0.115882635
                else:
                    if input[9460] < 0.38996992:
                        if input[8239] >= 0.10109547:
                            var181 = -0.045493703
                        else:
                            var181 = 0.076576866
                    else:
                        if input[6465] < 0.5095433:
                            var181 = 0.07581664
                        else:
                            var181 = -0.0014250893
    if input[2990] >= 0.18352535:
        if input[8466] < 0.14605875:
            var182 = -0.03434671
        else:
            var182 = -0.12804314
    else:
        if input[8424] < 0.13079146:
            var182 = 0.119400315
        else:
            if input[3916] < 0.094050944:
                if input[316] < 0.13926071:
                    var182 = -0.07443244
                else:
                    if input[5113] >= 0.112224616:
                        var182 = -0.074994855
                    else:
                        if input[9307] < 0.11914906:
                            var182 = -0.035484616
                        else:
                            var182 = 0.10221665
            else:
                if input[5378] < 0.18943173:
                    if input[4104] < 0.16645259:
                        var182 = 0.009837656
                    else:
                        if input[28] < 0.24747406:
                            var182 = 0.020679493
                        else:
                            var182 = 0.10899339
                else:
                    if input[9006] < 0.6959612:
                        var182 = 0.09750935
                    else:
                        if input[2104] >= 0.21082489:
                            var182 = 0.09862014
                        else:
                            var182 = -0.0016314861
    if input[2674] < 1.8983641:
        var183 = 0.15338837
    else:
        if input[5990] < 1.0:
            var183 = 0.11334087
        else:
            if input[5869] >= 0.21954967:
                var183 = -0.097759746
            else:
                if input[3891] < 1.3429911:
                    var183 = 0.14697051
                else:
                    if input[5372] < 1.2109509:
                        if input[5280] < 0.09770191:
                            var183 = 0.055622764
                        else:
                            var183 = 0.13505387
                    else:
                        if input[5998] < 0.56336385:
                            var183 = 0.09644448
                        else:
                            var183 = -0.0009991869
    if input[382] < 1.0851319:
        var184 = 0.20018034
    else:
        if input[4068] >= 0.15810794:
            if input[4076] < 0.29521683:
                var184 = 0.032114074
            else:
                if input[531] >= 0.17261451:
                    var184 = 0.013480415
                else:
                    if input[8626] >= 0.13634428:
                        var184 = 0.016125051
                    else:
                        if input[6851] >= 0.32578707:
                            var184 = 0.013946894
                        else:
                            var184 = -0.08933351
        else:
            if input[908] < 1.8546642:
                if input[908] < 0.37637743:
                    var184 = 0.15069626
                else:
                    var184 = 0.04858728
            else:
                if input[6310] >= 0.2118919:
                    if input[6312] < 0.3387697:
                        if input[3000] < 0.18728551:
                            var184 = 0.06225692
                        else:
                            var184 = -0.04112157
                    else:
                        if input[6785] < 0.11077283:
                            var184 = 0.02269952
                        else:
                            var184 = 0.16006409
                else:
                    if input[4878] >= 0.2709588:
                        if input[5280] >= 0.18003662:
                            var184 = -0.01613738
                        else:
                            var184 = -0.11633574
                    else:
                        if input[4269] < 0.36174938:
                            var184 = 0.10815531
                        else:
                            var184 = -0.00010588294
    if input[7930] < 1.4311116:
        var185 = -0.11726838
    else:
        if input[2658] < 1.2680178:
            if input[6785] < 0.18139964:
                if input[5113] < 0.15803108:
                    var185 = -0.0031453036
                else:
                    var185 = 0.12807404
            else:
                if input[2658] < 0.18789427:
                    if input[8239] >= 0.045043744:
                        var185 = -0.00028364974
                    else:
                        var185 = 0.110655956
                else:
                    if input[2658] < 0.24204013:
                        if input[2658] < 0.21385922:
                            var185 = -0.002766543
                        else:
                            var185 = -0.08268252
                    else:
                        if input[2658] < 0.2611085:
                            var185 = 0.08109975
                        else:
                            var185 = -0.015809026
        else:
            if input[6534] >= 0.14375475:
                if input[5280] >= 0.080951884:
                    if input[6534] < 0.30454513:
                        if input[3916] >= 0.13167988:
                            var185 = -0.06456935
                        else:
                            var185 = 0.033632234
                    else:
                        var185 = -0.080566786
                else:
                    if input[8466] >= 0.14092284:
                        if input[8466] < 0.17004934:
                            var185 = 0.08427085
                        else:
                            var185 = -0.018376233
                    else:
                        if input[8590] < 0.16432169:
                            var185 = 0.0466399
                        else:
                            var185 = -0.08822652
            else:
                if input[2680] < 2.00001:
                    if input[4666] < 0.15696235:
                        var185 = -0.013902281
                    else:
                        if input[8239] >= 0.13896155:
                            var185 = 0.0030867525
                        else:
                            var185 = 0.109140694
                else:
                    if input[5860] >= 0.14867797:
                        if input[5860] < 0.24390705:
                            var185 = -0.045085534
                        else:
                            var185 = -0.0950086
                    else:
                        if input[7334] < 0.53020924:
                            var185 = 0.04332158
                        else:
                            var185 = -0.00015829048
    if input[9274] >= 0.21646078:
        var186 = -0.09675154
    else:
        if input[8950] >= 0.26913023:
            if input[8950] < 0.5226389:
                var186 = -0.1293789
            else:
                var186 = -0.044271942
        else:
            if input[4466] >= 0.16465318:
                if input[5531] < 0.17541216:
                    var186 = 0.06186549
                else:
                    if input[3916] >= 0.09497498:
                        var186 = 0.0042907223
                    else:
                        if input[8239] >= 0.11828072:
                            var186 = 0.022533258
                        else:
                            var186 = -0.104882896
            else:
                if input[4869] >= 0.18953216:
                    if input[6785] >= 0.09426674:
                        var186 = 0.022569384
                    else:
                        if input[4869] < 0.26765487:
                            var186 = -0.12855719
                        else:
                            var186 = -0.061808605
                else:
                    if input[8236] < 0.42157084:
                        if input[5113] >= 0.14065382:
                            var186 = -0.055728562
                        else:
                            var186 = 0.11712664
                    else:
                        if input[3916] >= 0.2317871:
                            var186 = -0.067699365
                        else:
                            var186 = 0.0019822381
    if input[7518] < 1.2643865:
        if input[7518] < 0.30913737:
            var187 = 0.062697396
        else:
            var187 = 0.19273387
    else:
        if input[5459] < 0.2726077:
            if input[3844] >= 0.1614881:
                var187 = -0.01871792
            else:
                if input[5496] < 0.17010093:
                    var187 = 0.012008152
                else:
                    if input[9679] >= 0.11706728:
                        var187 = 0.036694027
                    else:
                        var187 = 0.11949245
        else:
            if input[5476] < 1.171566:
                if input[3432] >= 0.19938447:
                    var187 = -0.08764137
                else:
                    if input[5485] < 1.1764518:
                        var187 = -0.09112841
                    else:
                        if input[5494] < 1.0953032:
                            var187 = -0.08640799
                        else:
                            var187 = 0.05567897
            else:
                if input[3844] < 1.1679463:
                    if input[9679] < 0.23148172:
                        if input[9916] < 0.24337892:
                            var187 = 0.048330918
                        else:
                            var187 = -0.049372476
                    else:
                        if input[9916] >= 0.1336161:
                            var187 = -0.09691738
                        else:
                            var187 = 0.0395351
                else:
                    if input[3892] < 0.58796895:
                        var187 = 0.1665346
                    else:
                        if input[6785] < 2.00001:
                            var187 = 0.01137435
                        else:
                            var187 = -0.006622252
    if input[8466] < 0.08764862:
        var188 = 0.10741111
    else:
        if input[4170] >= 0.21689074:
            if input[316] >= 0.17587571:
                if input[5280] < 0.13514891:
                    var188 = -0.08353942
                else:
                    if input[316] < 0.32117757:
                        if input[4170] < 0.23287599:
                            var188 = 0.01075605
                        else:
                            var188 = 0.09329664
                    else:
                        if input[4170] < 0.3048383:
                            var188 = -0.007826581
                        else:
                            var188 = -0.06096242
            else:
                if input[9679] < 0.19376124:
                    var188 = -0.052924365
                else:
                    if input[5280] < 0.16537653:
                        if input[4224] >= 0.3775997:
                            var188 = -0.038411897
                        else:
                            var188 = 0.110042624
                    else:
                        if input[4224] < 1.6434999:
                            var188 = 0.10545119
                        else:
                            var188 = -0.014896682
        else:
            if input[5280] >= 0.11942284:
                if input[4326] >= 0.36764124:
                    var188 = 0.13935246
                else:
                    if input[5357] < 0.26583523:
                        if input[5280] < 0.121663675:
                            var188 = 0.025363052
                        else:
                            var188 = 0.1820639
                    else:
                        if input[4250] >= 0.46096686:
                            var188 = 0.17947295
                        else:
                            var188 = -0.031046158
            else:
                if input[5779] >= 0.21072488:
                    if input[5280] >= 0.103709504:
                        if input[5280] < 0.11113625:
                            var188 = 0.0029116631
                        else:
                            var188 = 0.061417628
                    else:
                        if input[5779] < 0.5037751:
                            var188 = -0.10301173
                        else:
                            var188 = 0.0069925375
                else:
                    if input[8042] >= 0.2392167:
                        if input[8042] < 0.39378986:
                            var188 = -0.10703679
                        else:
                            var188 = -0.017043836
                    else:
                        if input[8950] >= 0.26913023:
                            var188 = -0.12747453
                        else:
                            var188 = 0.0024333787
    if input[2371] < 1.2550216:
        if input[9491] >= 0.11910217:
            var189 = -0.010317026
        else:
            var189 = 0.13052933
    else:
        if input[1779] < 0.42820212:
            if input[1781] >= 0.28825215:
                var189 = 0.030176653
            else:
                var189 = 0.13527885
        else:
            if input[6418] < 0.34574562:
                if input[9307] < 0.16416317:
                    if input[6418] < 0.20892265:
                        var189 = -0.07310081
                    else:
                        var189 = -0.017864965
                else:
                    if input[3214] >= 0.24440584:
                        var189 = -0.053563215
                    else:
                        if input[7081] >= 0.16598743:
                            var189 = -0.052903414
                        else:
                            var189 = 0.08338564
            else:
                if input[6611] < 0.29233888:
                    if input[8416] < 0.42093498:
                        var189 = -0.035088267
                    else:
                        if input[4666] >= 0.13394572:
                            var189 = -0.033382785
                        else:
                            var189 = 0.091104954
                else:
                    if input[6321] < 0.2527963:
                        if input[9379] < 0.16865161:
                            var189 = 0.09752272
                        else:
                            var189 = 0.018007543
                    else:
                        if input[1350] < 0.28775734:
                            var189 = 0.07028889
                        else:
                            var189 = -0.0026665195
    if input[9001] >= 0.23266466:
        if input[5280] >= 0.09236117:
            var190 = -0.016010234
        else:
            if input[3916] >= 0.12487771:
                var190 = -0.031627372
            else:
                var190 = -0.12256527
    else:
        if input[8178] < 0.9718113:
            var190 = -0.10666623
        else:
            if input[6775] >= 0.14890413:
                if input[5676] >= 0.22542231:
                    var190 = 0.0111775575
                else:
                    if input[4012] < 0.18415895:
                        var190 = 0.010077807
                    else:
                        if input[6785] >= 0.09955875:
                            var190 = -0.029939968
                        else:
                            var190 = -0.11131034
            else:
                if input[6310] < 0.5506499:
                    if input[6312] >= 0.24030782:
                        if input[3000] < 0.19259176:
                            var190 = 0.04193419
                        else:
                            var190 = -0.009024473
                    else:
                        if input[6785] < 0.11077283:
                            var190 = 0.025574356
                        else:
                            var190 = 0.14880718
                else:
                    if input[4878] >= 0.2709588:
                        if input[5280] >= 0.18003662:
                            var190 = -0.01392479
                        else:
                            var190 = -0.11273998
                    else:
                        if input[1314] < 1.1838118:
                            var190 = 0.19710712
                        else:
                            var190 = 0.00101256
    if input[9048] < 1.1998109:
        var191 = 0.15817374
    else:
        if input[5531] >= 0.22028077:
            if input[6785] >= 0.1733538:
                var191 = 0.08501827
            else:
                if input[2957] >= 0.1438157:
                    var191 = 0.02594845
                else:
                    if input[6785] < 0.11985802:
                        var191 = 0.003533804
                    else:
                        if input[5280] < 0.14704065:
                            var191 = -0.016388115
                        else:
                            var191 = -0.097011514
        else:
            if input[710] >= 0.21040188:
                var191 = 0.08906236
            else:
                if input[595] >= 0.23258692:
                    if input[5533] < 0.44725674:
                        var191 = 0.003570875
                    else:
                        if input[8239] >= 0.11276287:
                            var191 = 0.025861561
                        else:
                            var191 = 0.13854547
                else:
                    if input[7635] >= 0.16517425:
                        if input[4170] >= 0.16479081:
                            var191 = 0.080234535
                        else:
                            var191 = -0.07365785
                    else:
                        if input[5935] < 0.9499705:
                            var191 = 0.16160452
                        else:
                            var191 = 0.000060788956
    if input[5113] < 0.100534074:
        var192 = 0.120856896
    else:
        if input[3270] < 0.48053354:
            var192 = 0.09668367
        else:
            if input[963] >= 0.26884383:
                if input[5280] >= 0.11687931:
                    if input[963] < 0.34112546:
                        var192 = -0.026207332
                    else:
                        var192 = 0.087033056
                else:
                    var192 = 0.13496919
            else:
                if input[1316] < 0.41825908:
                    var192 = 0.11119433
                else:
                    if input[8148] < 1.5330952:
                        var192 = 0.16342968
                    else:
                        if input[6978] < 0.35193887:
                            var192 = 0.09947773
                        else:
                            var192 = -0.0018198763
    if input[5280] < 0.048058297:
        var193 = 0.09844795
    else:
        if input[9094] < 0.61816126:
            var193 = 0.14004885
        else:
            if input[6319] < 0.23624277:
                var193 = 0.14285815
            else:
                if input[6386] < 0.51393956:
                    if input[5922] < 0.38865766:
                        var193 = 0.007608404
                    else:
                        var193 = -0.12839246
                else:
                    if input[275] < 0.47358245:
                        var193 = 0.11604947
                    else:
                        if input[5248] < 0.4452728:
                            var193 = 0.11299799
                        else:
                            var193 = -0.0010351589
    if input[892] >= 0.19968705:
        var194 = 0.1429592
    else:
        if input[6775] >= 0.14890413:
            if input[4012] < 0.18415895:
                var194 = 0.010062661
            else:
                if input[6785] >= 0.09955875:
                    var194 = -0.028476436
                else:
                    var194 = -0.10909126
        else:
            if input[8626] < 0.0879145:
                var194 = 0.11294333
            else:
                if input[4255] < 0.50742763:
                    if input[5694] >= 0.14700806:
                        if input[284] < 0.18539824:
                            var194 = -0.053308617
                        else:
                            var194 = 0.05518943
                    else:
                        if input[284] >= 0.103182234:
                            var194 = 0.034764804
                        else:
                            var194 = 0.15210323
                else:
                    if input[6078] >= 0.13189632:
                        var194 = -0.11135061
                    else:
                        if input[8398] < 0.12340101:
                            var194 = 0.11082073
                        else:
                            var194 = -0.00052828406
    if input[3413] < 2.00001:
        var195 = 0.15146756
    else:
        if input[1783] >= 0.25000238:
            var195 = 0.18733792
        else:
            if input[5736] >= 0.18982887:
                if input[3000] >= 0.17529456:
                    var195 = 0.009185773
                else:
                    if input[5113] >= 0.17151158:
                        var195 = 0.015971983
                    else:
                        var195 = 0.100240126
            else:
                if input[2876] >= 0.19514008:
                    if input[5280] >= 0.09800813:
                        if input[5280] < 0.11987202:
                            var195 = 0.08968673
                        else:
                            var195 = -0.009698574
                    else:
                        if input[6785] >= 0.11370214:
                            var195 = -0.00244789
                        else:
                            var195 = -0.12694456
                else:
                    if input[2679] < 1.4071828:
                        if input[5531] >= 0.15608986:
                            var195 = -0.03577333
                        else:
                            var195 = 0.08991038
                    else:
                        if input[8119] >= 0.20026378:
                            var195 = -0.08438824
                        else:
                            var195 = -0.00022103036
    if input[9119] < 1.3829981:
        var196 = -0.12122786
    else:
        if input[6306] < 0.60670716:
            var196 = -0.096051395
        else:
            if input[8083] >= 0.12325554:
                if input[3825] < 0.21786834:
                    var196 = -0.020674467
                else:
                    var196 = -0.13302395
            else:
                if input[5469] < 0.3876029:
                    if input[5496] < 0.17906852:
                        var196 = 0.011177182
                    else:
                        if input[3844] >= 0.07933914:
                            var196 = 0.018099738
                        else:
                            var196 = 0.12034775
                else:
                    if input[2387] < 0.6178757:
                        if input[9679] < 0.31660044:
                            var196 = 0.015257562
                        else:
                            var196 = 0.113303855
                    else:
                        if input[3087] >= 0.22436564:
                            var196 = -0.07863944
                        else:
                            var196 = 0.0008706486
    if input[4513] >= 0.23678838:
        if input[809] < 0.21000284:
            var197 = 0.09577819
        else:
            var197 = 0.025767246
    else:
        if input[4952] < 0.85737866:
            var197 = 0.13599087
        else:
            if input[3371] < 1.1404184:
                if input[5997] >= 0.10945203:
                    var197 = -0.08513143
                else:
                    if input[7376] < 0.2658061:
                        var197 = -0.09385587
                    else:
                        if input[2957] < 0.21118052:
                            var197 = -0.07158297
                        else:
                            var197 = 0.053655513
            else:
                if input[7518] < 1.2643865:
                    var197 = 0.1842058
                else:
                    if input[4590] < 1.0866749:
                        if input[4590] < 0.2138923:
                            var197 = 0.14976381
                        else:
                            var197 = 0.065391436
                    else:
                        if input[5990] < 1.0:
                            var197 = 0.11828249
                        else:
                            var197 = -0.0024612888
    if input[718] < 0.48527673:
        if input[718] < 0.28119665:
            if input[8239] < 0.17059773:
                var198 = 0.06947123
            else:
                var198 = -0.016292017
        else:
            var198 = 0.13623576
    else:
        if input[7318] >= 0.18262754:
            var198 = -0.092141286
        else:
            if input[3908] < 0.89272755:
                if input[8239] < 0.1569696:
                    var198 = -0.018982066
                else:
                    if input[6785] >= 0.08635712:
                        var198 = -0.017320883
                    else:
                        if input[5476] >= 0.17067823:
                            var198 = 0.0049439725
                        else:
                            var198 = 0.14245148
            else:
                if input[9525] < 1.241941:
                    var198 = 0.11601142
                else:
                    if input[1125] < 0.2844278:
                        if input[4012] < 0.14068377:
                            var198 = -0.04372319
                        else:
                            var198 = 0.09214401
                    else:
                        if input[911] >= 0.2729628:
                            var198 = 0.16495766
                        else:
                            var198 = -0.0011732436
    if input[425] >= 0.34232843:
        var199 = 0.14865802
    else:
        if input[2104] >= 0.21082489:
            var199 = 0.09734226
        else:
            if input[6404] < 1.1448104:
                var199 = 0.1140752
            else:
                if input[749] < 0.13306631:
                    var199 = 0.11834586
                else:
                    if input[4170] < 1.705911:
                        if input[316] >= 0.11482531:
                            var199 = -0.03300185
                        else:
                            var199 = 0.049891215
                    else:
                        if input[3000] < 1.0769:
                            var199 = 0.043177363
                        else:
                            var199 = -0.003714174
    if input[2782] >= 0.16067313:
        if input[9679] < 0.11667014:
            var200 = 0.01732835
        else:
            if input[4170] >= 0.13133019:
                var200 = -0.043202784
            else:
                if input[5280] < 0.22088154:
                    var200 = -0.10727926
                else:
                    var200 = -0.028532062
    else:
        if input[4908] < 2.00001:
            var200 = -0.11121335
        else:
            if input[8792] >= 0.13564949:
                if input[6785] >= 0.10403197:
                    var200 = 0.0095468955
                else:
                    if input[8792] < 0.19012164:
                        var200 = -0.094865225
                    else:
                        var200 = -0.02993055
            else:
                if input[1248] >= 0.27413425:
                    var200 = 0.15000084
                else:
                    if input[5779] >= 0.21072488:
                        if input[5280] >= 0.103709504:
                            var200 = -0.0147407055
                        else:
                            var200 = -0.08870105
                    else:
                        if input[5025] >= 0.3102896:
                            var200 = 0.14631608
                        else:
                            var200 = 0.0013805634
    if input[5819] < 0.11763396:
        var201 = 0.12079048
    else:
        if input[3886] < 0.6754467:
            var201 = 0.13971649
        else:
            if input[5934] < 0.9473677:
                var201 = 0.14437525
            else:
                if input[5998] < 0.56336385:
                    var201 = 0.09537862
                else:
                    if input[9006] < 0.6959612:
                        var201 = 0.09612256
                    else:
                        if input[5997] >= 0.14992633:
                            var201 = -0.08669
                        else:
                            var201 = -0.00041919286
    if input[4666] < 0.08900275:
        var202 = 0.12032684
    else:
        if input[881] < 1.0:
            if input[9679] >= 0.13058922:
                var202 = 0.03517724
            else:
                var202 = 0.11680466
        else:
            if input[962] < 0.51143813:
                var202 = 0.104370765
            else:
                if input[3504] < 1.4315313:
                    if input[6785] < 0.16000375:
                        if input[5280] < 0.10509737:
                            var202 = 0.016056797
                        else:
                            var202 = -0.032445237
                    else:
                        if input[6213] >= 0.32411885:
                            var202 = 0.0036277154
                        else:
                            var202 = 0.11496408
                else:
                    if input[1347] < 0.27811673:
                        if input[6785] < 0.10372138:
                            var202 = 0.016276516
                        else:
                            var202 = 0.13801166
                    else:
                        if input[5422] >= 0.1981499:
                            var202 = -0.08629686
                        else:
                            var202 = -0.0010618054
    var203 = sigmoid(var147 + var148 + var149 + var150 + var151 + var152 + var153 + var154 + var155 + var156 + var157 + var158 + var159 + var160 + var161 + var162 + var163 + var164 + var165 + var166 + var167 + var168 + var169 + var170 + var171 + var172 + var173 + var174 + var175 + var176 + var177 + var178 + var179 + var180 + var181 + var182 + var183 + var184 + var185 + var186 + var187 + var188 + var189 + var190 + var191 + var192 + var193 + var194 + var195 + var196 + var197 + var198 + var199 + var200 + var201 + var202)
    return [1.0 - var203, var203]
