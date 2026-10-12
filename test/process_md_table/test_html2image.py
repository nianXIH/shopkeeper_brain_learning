from html2image import Html2Image

hti = Html2Image()

html_content = """
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; font-family:Arial;">
<tr><td></td><td>输入量程</td></tr>
<tr><td>功能</td><td>最大输入</td></tr>
<tr><td>交/直流电压</td><td>直流/交流电压600V</td></tr>
<tr><td>直流/交流电压</td><td>直流/交流电压600V, 200Vrms 用于200mV量程</td></tr>
<tr><td>mA直流</td><td>200mA 250V快速熔断保险丝</td></tr>
<tr><td>A DC</td><td>10A 250V 快速熔断保险丝(最多每15分钟,需时30秒)</td></tr>
<tr><td>电阻,短路测试</td><td>250Vrms, 最多15秒</td></tr>
</table>
"""

hti.screenshot(html_str=html_content, save_as="table.png", size=(800, 400))