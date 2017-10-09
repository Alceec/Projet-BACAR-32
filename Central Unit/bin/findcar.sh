#!/bin/bash

if [ "$(id -u)" != "0" ]; then
        echo "Vous devez executer cette fonction en root/avec sudo !"
        exit 1
fi
#récupère l'ip locale/réseau
#iplocal=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')
if [ -z "$2" ]; then
	iplocal="192.168.1.0/23"
	if ping -c 1 -W 1 192.168.42.1 &> /dev/null
	then
		echo "Vous êtes connecté en wifi sur une bacar dont l'IP est 192.168.42.1."
		sed -i "3s/.*/192.168.42.1  bacar /" /etc/hosts
		exit 1
	fi
else
	iplocal="$2"
fi

#numéro de la voiture recherchée
nb=$(printf %02d $1)
if [ "$nb" -eq 00 ]; then
	nb=""
fi
#cherche pour toutes les bacar sur le réseau local
echo "Recherche de voitures sur les locaux 192.168.0.0/24 et 192.168.1.0/24. Veuillez patienter, cette recherche peut prendre du temps !"
IPBACAR=$(nmap -sP -PS22 $iplocal | grep bacar$nb | cut -d "(" -f2 | cut -d ")" -f1)
ipbacars=($IPBACAR)
nbcars=${#ipbacars[@]}
if [ "$nbcars" -gt "1" ]; then
	echo "$nbcars voitures ont été trouvées sur le réseau. Veuillez spécifier le numéro de votre voiture à la fin de la commande findcar.sh (p/ex : sudo findcar.sh 2 pour la voiture 2)"
elif [ "$nbcars" -eq "0" ]; then
	echo "Aucune voiture n'a été trouvée sur les réseaux 192.168.0.0/24 ou 192.168.1.1/24. Veuillez vérifier en priorité que votre voiture est connectée sur le même réseau que votre machine virtuelle. Si tel est en le cas veuillez spécifier le réseau local de votre router avec la commande 'sudo findcard.sh nb plage_réseau' où plage_réseau est la plage d'attribution de votre router et nb le numéro de votre voiture (p/ex : sudo findcar.sh 2 10.0.0.0/24 pour la voiture 2 sur le réseau 10.0.0.0) !"
else
	echo "Une voiture a été trouvée sur votre réseau avec l'IP $IPBACAR."
	sed -i "3s/.*/$IPBACAR  bacar /" /etc/hosts
fi
