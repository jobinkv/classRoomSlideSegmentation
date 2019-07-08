import numpy as np
from subprocess import call

header="""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
div {
    margin: 2px;
border: 50px solid #200000;
background-color: lightblue;
}
img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
table {
    border-collapse: collapse;
    width: 100%;
}

th, td {
    text-align: left;
    padding: 8px;
}

tr:nth-child(even){background-color: #f2f2f2}
tr:nth-child(odd){background-color: #b0b0b0}

th {
    background-color: #000050;
    color: white;
}
</style>
</head>
<body>
<div>
<h1 align="center">The training progres</h1>
"""
imgHead = '<img src="'
imgTail='" alt="curve" style="width:50%;">'
tablehead1="""
<table>
  <tr>
<th> Iteration:</th>
<th> Background</th>
<th> Headings </th>
<th> List </th>
<th> Paragraph </th>
<th> Figure</th>
<th> Table </th>
<th> Caption</th>
<th> Equation</th>
<th> Mean</th>
  </tr>
"""
tableTail="</table>"
tail="""
</div>
</body>
</html>
"""

def getHtmlTable(sep_iu,args):
        out=' '
        for item in sep_iu:
                #out=out+'<tr><td><a href=./'+args['jobid']+'/'+str(item[0])+'>'+ str(item[0])+'</a></td>'
                out=out+'<tr><td><a href=/cgi-bin/slide_view/s4m.py?exp='+args['jobid']+'&it='+str(item[0])+'>'+ str(item[0])+'</a></td>'
                for i in range(1,len(item)):
                        out=out+'<td> '+str(round(item[i]*100, 2))+'%</td>'
                out=out+'</tr>'
        return out

def paramPrint(args):
        out='<table><tr><th> Parameters</th><th> Values</th></tr>'
        for key, value in args.items() :
                if key=='best_record':
                        continue
                out=out+'<tr><td>'+key+'</td>'+'<td>'+str(value)+'</td></tr>'
        return out+'</table>'
def logHtml(val,test,args):
        file1 = open(args['jobid']+'.html','w')
        file1.write(header)
        file1.write(imgHead)
        file1.write(args['jobid']+'.png')
        file1.write(imgTail)
        file1.write('<h2 align="center">Experiment Parameters</h2>')
        file1.write(paramPrint(args))
        file1.write('<h2 align="center">Validation IoU</h2>')
        file1.write(tablehead1)
        file1.write(getHtmlTable(val,args))
        file1.write(tableTail)
        file1.write('<h2 align="center">Test IoU</h2>')
        file1.write(tablehead1)
        file1.write(getHtmlTable(test,args))
        file1.write(tableTail)
        file1.write(tail)
        file1.close()
        call(["scp", args['jobid']+".html", "jobinkv@10.2.16.142:/home/jobinkv/Documents/r1/19wavc/"])
        print ('To view the training progres, please visit')
        print ('http://10.2.16.142/r1/19wavc/'+args['jobid']+'.html')
